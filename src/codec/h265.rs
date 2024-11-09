//! [H.265]-encoded video.

use std::convert::TryFrom;
use std::fmt::Write;

use base64::Engine;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use hevc_parser::hevc::{NalHeader, UnitType};
use log::{debug, log_enabled, trace};

use crate::rtp::ReceivedPacket;

use super::VideoFrame;

#[derive(Debug)]
pub(crate) struct Depacketizer {
    input_state: DepacketizerInputState,

    /// A complete video frame ready for pull.
    pending: Option<VideoFrame>,

    parameters: Option<InternalParameters>,

    /// In state `PreMark`, pieces of NALs, excluding their header bytes.
    /// Kept around (empty) in other states to re-use the backing allocation.
    pieces: Vec<Bytes>,

    /// In state `PreMark`, an entry for each NAL.
    /// Kept around (empty) in other states to re-use the backing allocation.
    nals: Vec<Nal>,
}

#[derive(Debug)]
struct Nal {
    hdr: NalHeader,

    /// The length of `Depacketizer::pieces` as this NAL finishes.
    next_piece_idx: u32,

    /// The total length of this NAL, including the header byte.
    len: u32,
}

/// An access unit that is currently being accumulated during `PreMark` state.
#[derive(Debug)]
struct AccessUnit {
    start_ctx: crate::PacketContext,
    end_ctx: crate::PacketContext,
    timestamp: crate::Timestamp,
    stream_id: usize,

    /// True iff currently processing a FU-A.
    in_fu: bool,

    /// RTP packets lost as this access unit was starting.
    loss: u16,

    same_ts_as_prev: bool,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum DepacketizerInputState {
    /// Not yet processing an access unit.
    New,

    /// Ignoring the remainder of an access unit because of interior packet loss.
    Loss {
        timestamp: crate::Timestamp,
        pkts: u16,
    },

    /// Currently processing an access unit.
    /// This will be flushed after a marked packet or when receiving a later timestamp.
    PreMark(AccessUnit),

    /// Finished processing the given packet. It's an error to receive the same timestamp again.
    PostMark {
        timestamp: crate::Timestamp,
        loss: u16,
    },
}

impl Depacketizer {
    pub(super) fn new(
        clock_rate: u32,
        format_specific_params: Option<&str>,
    ) -> Result<Self, String> {
        if clock_rate != 90_000 {
            return Err(format!(
                "invalid H.265 clock rate {clock_rate}; must always be 90000"
            ));
        }

        let parameters = match format_specific_params {
            None => None,
            Some(fp) => match InternalParameters::parse_format_specific_params(fp) {
                Ok(p) => Some(p),
                Err(e) => {
                    log::warn!("Ignoring bad H.265 format-specific-params {:?}: {}", fp, e);
                    None
                }
            },
        };
        Ok(Depacketizer {
            input_state: DepacketizerInputState::New,
            pending: None,
            pieces: Vec::new(),
            nals: Vec::new(),
            parameters,
        })
    }

    pub(super) fn parameters(&self) -> Option<super::ParametersRef> {
        self.parameters
            .as_ref()
            .map(|p| super::ParametersRef::Video(&p.generic_parameters))
    }

    pub(super) fn push(&mut self, pkt: ReceivedPacket) -> Result<(), String> {
        // Push shouldn't be called until pull is exhausted.
        if let Some(p) = self.pending.as_ref() {
            panic!("push with data already pending: {p:?}");
        }

        let mut access_unit =
            match std::mem::replace(&mut self.input_state, DepacketizerInputState::New) {
                DepacketizerInputState::New => {
                    debug_assert!(self.nals.is_empty());
                    debug_assert!(self.pieces.is_empty());
                    AccessUnit::start(&pkt, 0, false)
                }
                DepacketizerInputState::PreMark(mut access_unit) => {
                    let loss = pkt.loss();
                    if loss > 0 {
                        self.nals.clear();
                        self.pieces.clear();
                        if access_unit.timestamp.timestamp == pkt.timestamp().timestamp {
                            // Loss within this access unit. Ignore until mark or new timestamp.
                            self.input_state = if pkt.mark() {
                                DepacketizerInputState::PostMark {
                                    timestamp: pkt.timestamp(),
                                    loss,
                                }
                            } else {
                                self.pieces.clear();
                                self.nals.clear();
                                DepacketizerInputState::Loss {
                                    timestamp: pkt.timestamp(),
                                    pkts: loss,
                                }
                            };
                            return Ok(());
                        }
                        // A suffix of a previous access unit was lost; discard it.
                        // A prefix of the new one may have been lost; try parsing.
                        AccessUnit::start(&pkt, 0, false)
                    } else if access_unit.timestamp.timestamp != pkt.timestamp().timestamp {
                        if access_unit.in_fu {
                            return Err(format!(
                                "Timestamp changed from {} to {} in the middle of a fragmented NAL",
                                access_unit.timestamp,
                                pkt.timestamp()
                            ));
                        }
                        let last_nal_hdr = self
                            .nals
                            .last()
                            .ok_or("nals should not be empty".to_string())?
                            .hdr;
                        if !access_unit.in_fu && can_end_au(last_nal_hdr.nal_unit_type()) {
                            access_unit.end_ctx = *pkt.ctx();
                            self.pending =
                                Some(self.finalize_access_unit(access_unit, "ts change")?);
                            AccessUnit::start(&pkt, 0, false)
                        } else {
                            log::debug!(
                                "Bogus mid-access unit timestamp change after {:?}",
                                last_nal_hdr
                            );
                            access_unit.timestamp.timestamp = pkt.timestamp().timestamp;
                            access_unit
                        }
                    } else {
                        access_unit
                    }
                }
                DepacketizerInputState::PostMark {
                    timestamp: state_ts,
                    loss,
                } => {
                    debug_assert!(self.nals.is_empty());
                    debug_assert!(self.pieces.is_empty());
                    AccessUnit::start(&pkt, loss, state_ts.timestamp == pkt.timestamp().timestamp)
                }
                DepacketizerInputState::Loss {
                    timestamp,
                    mut pkts,
                } => {
                    debug_assert!(self.nals.is_empty());
                    debug_assert!(self.pieces.is_empty());
                    if pkt.timestamp().timestamp == timestamp.timestamp {
                        pkts += pkt.loss();
                        self.input_state = DepacketizerInputState::Loss { timestamp, pkts };
                        return Ok(());
                    }
                    AccessUnit::start(&pkt, pkts, false)
                }
            };

        let ctx = *pkt.ctx();
        let mark = pkt.mark();
        let loss = pkt.loss();
        let timestamp = pkt.timestamp();
        let mut data = pkt.into_payload_bytes();
        if data.is_empty() {
            return Err("Empty NAL".into());
        }

        let nal_header = data.get_u16();
        let hdr = NalHeader::new(nal_header)
            .map_err(|_| format!("NAL header {nal_header:02x} has F bit set"))?;
        match hdr.nal_unit_type().id() {
            1..=47 => {
                // Single NAL Unit. https://datatracker.ietf.org/doc/html/rfc7798#section-4.4.1
                if access_unit.in_fu {
                    return Err(format!(
                        "Non-fragmented NAL {nal_header:02x} while fragment in progress"
                    ));
                }
                let len =
                    u32::try_from(data.len()).map_err(|_| "data len > u16::MAX".to_string())? + 2;
                let next_piece_idx = self.add_piece(data)?;
                self.nals.push(Nal {
                    hdr,
                    next_piece_idx,
                    len,
                });
            }
            48 => {
                // Aggregation Packet. https://datatracker.ietf.org/doc/html/rfc7798#section-4.4.2
                loop {
                    if data.remaining() < 3 {
                        return Err(format!(
                            "AP has {} remaining bytes; expecting 2-byte length, non-empty NAL",
                            data.remaining()
                        ));
                    }
                    let len = data.get_u16();
                    if len == 0 {
                        return Err("zero length in AP".into());
                    }
                    match data.remaining().cmp(&usize::from(len)) {
                        std::cmp::Ordering::Less => {
                            return Err(format!(
                                "AP too short: {} bytes remaining, expecting {}-byte NAL",
                                data.remaining(),
                                len
                            ))
                        }
                        std::cmp::Ordering::Equal => {
                            let nal_header = data.get_u16();
                            let hdr = NalHeader::new(nal_header).map_err(|_| {
                                format!("NAL header {nal_header:02x} has F bit set")
                            })?;
                            let next_piece_idx = self.add_piece(data)?;
                            self.nals.push(Nal {
                                hdr,
                                next_piece_idx,
                                len: u32::from(len),
                            });
                            break;
                        }
                        std::cmp::Ordering::Greater => {
                            let nal_header = data.get_u16();
                            let hdr = NalHeader::new(nal_header).map_err(|_| {
                                format!("NAL header {nal_header:02x} has F bit set")
                            })?;
                            let piece = data.split_to(usize::from(len));
                            let next_piece_idx = self.add_piece(piece)?;
                            self.nals.push(Nal {
                                hdr,
                                next_piece_idx,
                                len: u32::from(len),
                            });
                        }
                    }
                }
            }
            49 => {
                // Fragmentation Unit. https://datatracker.ietf.org/doc/html/rfc7798#section-4.4.3
                if data.len() < 2 {
                    return Err(format!("FU len {} too short", data.len()));
                }
                let fu_header = data.get_u8();
                let start = (fu_header & 0b10000000) != 0;
                let end = (fu_header & 0b01000000) != 0;
                let fu_type = fu_header & 0b00111111;
                let hdr = NalHeader::new(
                    ((fu_type as u16) << 9)
                        | (hdr.nuh_layer_id() << 3) as u16
                        | hdr.nuh_temporal_id_plus1() as u16,
                )
                .map_err(|_| "NalHeader is invalid".to_string())?;
                if start && end {
                    return Err(format!("Invalid FU header {fu_header:02x}"));
                }
                if !end && mark {
                    return Err("FU pkt with MARK && !END".into());
                }
                let u32_len = u32::try_from(data.len())
                    .map_err(|_| "RTP packet len must be < u16::MAX".to_string())?;
                match (start, access_unit.in_fu) {
                    (true, true) => return Err("FU with start bit while frag in progress".into()),
                    (true, false) => {
                        self.add_piece(data)?;
                        self.nals.push(Nal {
                            hdr,
                            next_piece_idx: u32::MAX, // should be overwritten later.
                            len: 2 + u32_len,
                        });
                        access_unit.in_fu = true;
                    }
                    (false, true) => {
                        let pieces = self.add_piece(data)?;
                        let nal = self
                            .nals
                            .last_mut()
                            .ok_or("nals non-empty while in fu".to_string())?;
                        if u16::from(hdr) != u16::from(nal.hdr) {
                            return Err(format!(
                                "FU has inconsistent NAL type: {:?} then {:?}",
                                nal.hdr, hdr,
                            ));
                        }
                        nal.len += u32_len;
                        if end {
                            nal.next_piece_idx = pieces;
                            access_unit.in_fu = false;
                        } else if mark {
                            return Err("FU has MARK and no END".into());
                        }
                    }
                    (false, false) => {
                        if loss > 0 {
                            self.pieces.clear();
                            self.nals.clear();
                            self.input_state = DepacketizerInputState::Loss {
                                timestamp,
                                pkts: loss,
                            };
                            return Ok(());
                        }
                        return Err("FU has start bit unset while no frag in progress".into());
                    }
                }
            }
            _ => return Err(format!("unexpected/bad nal header {nal_header:02x}")),
        }

        self.input_state = if mark {
            let last_nal_hdr = self
                .nals
                .last()
                .ok_or("nals should not be empty after mark".to_string())?
                .hdr;
            if can_end_au(last_nal_hdr.nal_unit_type()) {
                access_unit.end_ctx = ctx;
                self.pending = Some(self.finalize_access_unit(access_unit, "mark")?);
                DepacketizerInputState::PostMark { timestamp, loss: 0 }
            } else {
                log::debug!(
                    "Bogus mid-access unit timestamp change after {:?}",
                    last_nal_hdr
                );
                access_unit.timestamp.timestamp = timestamp.timestamp;
                DepacketizerInputState::PreMark(access_unit)
            }
        } else {
            DepacketizerInputState::PreMark(access_unit)
        };
        Ok(())
    }

    pub(super) fn pull(&mut self) -> Option<super::CodecItem> {
        self.pending.take().map(super::CodecItem::VideoFrame)
    }

    /// Adds a piece to `self.pieces`, erroring if it becomes absurdly large.
    fn add_piece(&mut self, piece: Bytes) -> Result<u32, String> {
        self.pieces.push(piece);
        u32::try_from(self.pieces.len()).map_err(|_| "more than u32::MAX pieces!".to_string())
    }

    /// Logs information about each access unit.
    /// Currently, "bad" access units (violating certain specification rules)
    /// are logged at debug priority, and others are logged at trace priority.
    fn log_access_unit(&self, au: &AccessUnit, reason: &str) {
        let mut errs = String::new();
        if au.same_ts_as_prev {
            errs.push_str("\n* same timestamp as previous access unit");
        }
        if !errs.is_empty() {
            let mut nals = String::new();
            for (i, nal) in self.nals.iter().enumerate() {
                let _ = write!(&mut nals, "\n  {}: {:?}", i, nal.hdr);
            }
            debug!(
                "bad access unit (ended by {}) at ts {}\nerrors are:{}\nNALs are:{}",
                reason, au.timestamp, errs, nals
            );
        } else if log_enabled!(log::Level::Trace) {
            let mut nals = String::new();
            for (i, nal) in self.nals.iter().enumerate() {
                let _ = write!(&mut nals, "\n  {}: {:?}", i, nal.hdr);
            }
            trace!(
                "access unit (ended by {}) at ts {}; NALS are:{}",
                reason,
                au.timestamp,
                nals
            );
        }
    }

    fn finalize_access_unit(&mut self, au: AccessUnit, reason: &str) -> Result<VideoFrame, String> {
        let mut piece_idx = 0;
        let mut retained_len = 0usize;
        let mut is_random_access_point = false;
        let is_disposable = false;
        let mut new_vps = None::<Bytes>;
        let mut new_sps = None::<Bytes>;
        let mut new_pps = None::<Bytes>;

        if log_enabled!(log::Level::Debug) {
            self.log_access_unit(&au, reason);
        }
        for nal in &self.nals {
            let next_piece_idx = usize::try_from(nal.next_piece_idx).expect("u32 fits in usize");
            if next_piece_idx > self.pieces.len() {
                return Err("Incomplete buffered nals finalizing access unit".into());
            }
            let nal_pieces = &self.pieces[piece_idx..next_piece_idx];
            match nal.hdr.nal_unit_type() {
                UnitType::NalVps => {
                    if self
                        .parameters
                        .as_ref()
                        .map(|p| !nal_matches(&p.vps_nal[..], nal.hdr, nal_pieces))
                        .unwrap_or(true)
                    {
                        new_vps = Some(to_bytes(nal.hdr, nal.len, nal_pieces));
                    }
                }
                UnitType::NalSps => {
                    if self
                        .parameters
                        .as_ref()
                        .map(|p| !nal_matches(&p.sps_nal[..], nal.hdr, nal_pieces))
                        .unwrap_or(true)
                    {
                        new_sps = Some(to_bytes(nal.hdr, nal.len, nal_pieces));
                    }
                }
                UnitType::NalPps => {
                    if self
                        .parameters
                        .as_ref()
                        .map(|p| !nal_matches(&p.pps_nal[..], nal.hdr, nal_pieces))
                        .unwrap_or(true)
                    {
                        new_pps = Some(to_bytes(nal.hdr, nal.len, nal_pieces));
                    }
                }
                UnitType::NalIdrNLp
                | UnitType::NalIdrWRadl
                | UnitType::NalCraNut
                | UnitType::NalBlaNLp
                | UnitType::NalBlaWLp
                | UnitType::NalBlaWRadl => {
                    is_random_access_point = true;
                }
                _ => {}
            }
            retained_len += 4usize + usize::try_from(nal.len).expect("u32 fits in usize");
            piece_idx = next_piece_idx;
        }
        let mut nals = vec![];
        piece_idx = 0;
        for nal in &self.nals {
            let next_piece_idx = usize::try_from(nal.next_piece_idx).expect("u32 fits in usize");
            let nal_pieces = &self.pieces[piece_idx..next_piece_idx];

            nals.extend_from_slice(&[0, 0, 0, 1]);
            nals.push(((u16::from(nal.hdr) >> 8) & 0xFF) as u8);
            nals.push((u16::from(nal.hdr) & 0xFF) as u8);

            let mut actual_len = 2;
            for piece in nal_pieces {
                nals.extend_from_slice(&piece[..]);
                actual_len += piece.len();
            }
            debug_assert_eq!(
                usize::try_from(nal.len).expect("u32 fits in usize"),
                actual_len
            );
            piece_idx = next_piece_idx;
        }
        let mut parser = hevc_parser::HevcParser::default();
        let mut offsets = vec![];
        parser.get_offsets(&nals, &mut offsets);
        let parsed_nals = parser
            .split_nals(
                &nals,
                &offsets,
                *offsets
                    .last()
                    .ok_or("nal offsets should not be empty on finalize".to_string())?,
                false,
            )
            .map_err(|_| "failed to get nals from hevc_parser".to_string())?;

        let mut data = Vec::with_capacity(retained_len);
        for nal in parsed_nals {
            data.extend_from_slice(&((nal.end - nal.start) as u32).to_be_bytes()[..]);
            data.extend_from_slice(&nals[nal.start..nal.end]);
        }

        debug_assert_eq!(retained_len, data.len());

        self.nals.clear();
        self.pieces.clear();

        let all_new_params = new_vps.is_some() && new_sps.is_some() && new_pps.is_some();
        let some_new_params = new_vps.is_some() || new_sps.is_some() || new_pps.is_some();
        let has_new_parameters = if all_new_params || (some_new_params && self.parameters.is_some())
        {
            let vps_nal = new_vps
                .as_deref()
                .unwrap_or_else(|| &self.parameters.as_ref().unwrap().vps_nal);
            let sps_nal = new_sps
                .as_deref()
                .unwrap_or_else(|| &self.parameters.as_ref().unwrap().sps_nal);
            let pps_nal = new_pps
                .as_deref()
                .unwrap_or_else(|| &self.parameters.as_ref().unwrap().pps_nal);
            self.parameters = Some(InternalParameters::parse_vps_sps_pps(
                vps_nal, sps_nal, pps_nal,
            )?);
            true
        } else {
            false
        };

        Ok(VideoFrame {
            has_new_parameters,
            loss: au.loss,
            start_ctx: au.start_ctx,
            end_ctx: au.end_ctx,
            timestamp: au.timestamp,
            stream_id: au.stream_id,
            is_random_access_point,
            is_disposable,
            data,
        })
    }
}

/// Returns true if we allow the given NAL unit type to end an access unit.
fn can_end_au(nal_unit_type: UnitType) -> bool {
    nal_unit_type != UnitType::NalVps
        && nal_unit_type != UnitType::NalSps
        && nal_unit_type != UnitType::NalPps
}

impl AccessUnit {
    fn start(
        pkt: &crate::rtp::ReceivedPacket,
        additional_loss: u16,
        same_ts_as_prev: bool,
    ) -> Self {
        AccessUnit {
            start_ctx: *pkt.ctx(),
            end_ctx: *pkt.ctx(),
            timestamp: pkt.timestamp(),
            stream_id: pkt.stream_id(),
            in_fu: false,

            // TODO: overflow?
            loss: pkt.loss() + additional_loss,
            same_ts_as_prev,
        }
    }
}

#[derive(Clone, Debug)]
struct InternalParameters {
    generic_parameters: super::VideoParameters,

    /// The (single) VPS NAL.
    vps_nal: Bytes,

    /// The (single) SPS NAL.
    sps_nal: Bytes,

    /// The (single) PPS NAL.
    pps_nal: Bytes,
}

impl InternalParameters {
    /// Parses metadata from the `format-specific-params` of a SDP `fmtp` media attribute.
    fn parse_format_specific_params(format_specific_params: &str) -> Result<Self, String> {
        let mut sps_nal = None;
        let mut pps_nal = None;
        let mut vps_nal = None;
        for p in format_specific_params.split(';') {
            match p.trim().split_once('=') {
                Some((key, value)) => {
                    if key == "tx-mode" && value != "SRST" {
                        return Err(format!(
                            "unsupported/unexpected tx-mode {value}; expected SRST"
                        ));
                    }
                    if !matches!(key, "sprop-sps" | "sprop-pps" | "sprop-vps") {
                        continue;
                    }
                    let nal = base64::engine::general_purpose::STANDARD
                        .decode(value)
                        .map_err(|_| {
                            "bad parameter: NAL has invalid base64 encoding".to_string()
                        })?;
                    if nal.is_empty() {
                        return Err(format!("bad parameter {key}: empty NAL"));
                    }
                    match key {
                        "sprop-sps" => {
                            if sps_nal.is_none() {
                                sps_nal = Some(nal);
                            } else {
                                return Err("multiple SPSs".into());
                            }
                        }
                        "sprop-pps" => {
                            if pps_nal.is_none() {
                                pps_nal = Some(nal);
                            } else {
                                return Err("multiple PPSs".into());
                            }
                        }
                        "sprop-vps" => {
                            if vps_nal.is_none() {
                                vps_nal = Some(nal);
                            } else {
                                return Err("multiple VPSs".into());
                            }
                        }
                        _ => (),
                    }
                }
                None => return Err("key without value".into()),
            }
        }
        let vps_nal = vps_nal.ok_or_else(|| "no vps".to_string())?;
        let sps_nal = sps_nal.ok_or_else(|| "no sps".to_string())?;
        let pps_nal = pps_nal.ok_or_else(|| "no pps".to_string())?;

        Self::parse_vps_sps_pps(&vps_nal, &sps_nal, &pps_nal)
    }

    fn parse_vps_sps_pps(
        vps_nal: &[u8],
        sps_nal: &[u8],
        pps_nal: &[u8],
    ) -> Result<InternalParameters, String> {
        let sps_nal_bytes = hevc_parser::clear_start_code_emulation_prevention_3_byte(sps_nal);
        let mut reader =
            bitvec_helpers::bitstream_io_reader::BsIoVecReader::from_vec(sps_nal_bytes);
        reader
            .get_n::<u16>(16)
            .map_err(|_| "failed to read sps header".to_string())?;
        let sps = hevc_parser::hevc::sps::SPSNAL::parse(&mut reader)
            .map_err(|err| format!("failed to parse sps: {err}"))?;

        let pps_nal_bytes = hevc_parser::clear_start_code_emulation_prevention_3_byte(pps_nal);
        let mut reader =
            bitvec_helpers::bitstream_io_reader::BsIoVecReader::from_vec(pps_nal_bytes);
        reader
            .get_n::<u16>(16)
            .map_err(|_| "failed to read pps header".to_string())?;
        let pps = hevc_parser::hevc::pps::PPSNAL::parse(&mut reader)
            .map_err(|err| format!("failed to parse pps: {err}"))?;

        let general_profile_space = match sps.ptl.general_profile_space {
            0 => "",
            1 => "A",
            2 => "B",
            3 => "C",
            _ => return Err("expected general_profile_space to be 0 - 3".into()),
        };
        let general_profile_idc = sps.ptl.general_profile_idc;
        let mut general_profile_compatibility_flags = 0u32;
        for (idx, b) in sps
            .ptl
            .general_profile_compatibility_flag
            .iter()
            .rev()
            .enumerate()
        {
            general_profile_compatibility_flags |= ((*b as u32) << idx) as u32;
        }
        let general_tier_flag = match sps.ptl.general_tier_flag {
            true => "H",
            false => "L",
        };
        let general_level_idc = sps.ptl.general_level_idc;
        let gpsf = sps.ptl.general_progressive_source_flag as u8;
        let gisf = sps.ptl.general_interlaced_source_flag as u8;
        let gnpcf = sps.ptl.general_non_packed_constraint_flag as u8;
        let gfoc = sps.ptl.general_frame_only_constraint_flag as u8;
        let constraints = (gpsf << 7) | (gisf << 6) | (gnpcf << 5) | (gfoc << 4);
        let rfc6381_codec = format!(
            "hvc1.{general_profile_space}{general_profile_idc}.{general_profile_compatibility_flags:02X}.{general_tier_flag}{general_level_idc}.{constraints:02X}",
        );

        let pixel_dimensions = (sps.width() as u32, sps.height() as u32);
        let (pixel_aspect_ratio, frame_rate) = if sps.vui_present {
            let frame_rate = (
                sps.vui_parameters.vui_num_units_in_tick,
                sps.vui_parameters.vui_time_scale,
            );
            (
                sps.vui_parameters
                    .aspect_ratio()
                    .map(|(w, h)| (w as u32, h as u32)),
                Some(frame_rate),
            )
        } else {
            (None, None)
        };

        let mut hevc_decoder_config = BytesMut::new(); //with_capacity(45 + vps_nal.len() + sps_nal.len() + pps_nal.len());
        hevc_decoder_config.put_u8(1); // configurationVersion

        let general_profile_space = (sps.ptl.general_profile_space as u8) << 6;
        let general_tier_flag = (sps.ptl.general_tier_flag as u8) << 5;
        let general_profile_idc = (sps.ptl.general_profile_idc as u8) & 0x1F;
        hevc_decoder_config.put_u8(general_profile_space | general_tier_flag | general_profile_idc); // general_profile_space(2 bits) / general_tier_flag(1 bit) / general_profile_idc(5 bits)

        hevc_decoder_config.put_u32(general_profile_compatibility_flags); // general_profile_compatibility_flags

        hevc_decoder_config.put_u8(constraints); // general_constraint_indicator_flags
        hevc_decoder_config.put_u32(0); // general_constraint_indicator_flags
        hevc_decoder_config.put_u8(0); // general_constraint_indicator_flags

        hevc_decoder_config.put_u8(general_level_idc); // general_level_idc
        hevc_decoder_config
            .put_u16((0xF000 | (0xFFFF & sps.vui_parameters.min_spatial_segmentation_idc)) as u16); // reserved(4 bits) / min_spatial_segmentation_idc(12 bits)
        let parallelism_type: u8 = if sps.vui_parameters.min_spatial_segmentation_idc == 0 {
            0
        } else {
            if pps.entropy_coding_sync_enabled_flag && pps.tiles_enabled_flag {
                0
            } else if pps.entropy_coding_sync_enabled_flag {
                3
            } else if pps.tiles_enabled_flag {
                2
            } else {
                1
            }
        };
        hevc_decoder_config.put_u8(0b1111_1100 | parallelism_type); // reserved(6 bits) / parallelismType(2 bits)
        hevc_decoder_config.put_u8((0b1111_1100 | sps.chroma_format_idc) as u8); // reserved(6 bits) / chromaFormat(2 bits)
        hevc_decoder_config.put_u8((0b1111_1000 | (sps.bit_depth - 8)) as u8); // reserved(5 bits) / bitDepthLumaMinus8(3 bits)
        hevc_decoder_config.put_u8((0b1111_1000 | (sps.bit_depth_chroma - 8)) as u8); // reserved(5 bits) / bitDepthChromaMinus8(3 bits)
        hevc_decoder_config.put_u16(0); // avgFrameRate
        let max_sub_layers = 0b0011_1000 & sps.max_sub_layers << 3;
        let temporal_id_nesting_flag = 0b0000_0100 & ((sps.temporal_id_nesting_flag as u8) << 2);
        hevc_decoder_config.put_u8(max_sub_layers | temporal_id_nesting_flag | 0b0011); // constantFrameRate(2 bits) / numTemporalLayers(3 bits) / temporalIdNested(1 bit) / lengthSizeMinusOne(2 bits)
        hevc_decoder_config.put_u8(3); // numOfArrays

        // assuming 3 numArrays of numNalus of length 1. i.e. [[vps], [sps], [pps]]
        // vps
        hevc_decoder_config.put_u8(0b1010_0000); // array_completeness(1 bit) / reserved(1 bit) / NAL_unit_type(6 bits)
        hevc_decoder_config.put_u16(1); // numNalus
        hevc_decoder_config.extend(
            &u16::try_from(vps_nal.len())
                .map_err(|_| format!("VPS NAL is {} bytes long; must fit in u16", sps_nal.len()))?
                .to_be_bytes()[..],
        ); // vps nalUnitLength
        hevc_decoder_config.extend_from_slice(vps_nal); // vps nalUnit
        hevc_decoder_config.put_u8(0b1010_0001); // array_completeness(1 bit) / reserved(1 bit) / NAL_unit_type(6 bits)
        hevc_decoder_config.put_u16(1); // numNalus
        hevc_decoder_config.extend(
            &u16::try_from(sps_nal.len())
                .map_err(|_| format!("SPS NAL is {} bytes long; must fit in u16", sps_nal.len()))?
                .to_be_bytes()[..],
        ); // sps nalUnitLength
        hevc_decoder_config.extend_from_slice(sps_nal); // sps nalUnit
        hevc_decoder_config.put_u8(0b1010_0010); // array_completeness(1 bit) / reserved(1 bit) / NAL_unit_type(6 bits)
        hevc_decoder_config.put_u16(1); // numNalus
        hevc_decoder_config.extend(
            &u16::try_from(pps_nal.len())
                .map_err(|_| format!("PPS NAL is {} bytes long; must fit in u16", pps_nal.len()))?
                .to_be_bytes()[..],
        ); // pps nalUnitLength
        hevc_decoder_config.extend_from_slice(pps_nal); // pps nalUnit

        let hevc_decoder_config = hevc_decoder_config.freeze();
        Ok(InternalParameters {
            generic_parameters: super::VideoParameters {
                rfc6381_codec,
                pixel_dimensions,
                pixel_aspect_ratio,
                frame_rate,
                extra_data: hevc_decoder_config,
            },
            vps_nal: Bytes::from(vps_nal.to_vec()),
            sps_nal: Bytes::from(sps_nal.to_vec()),
            pps_nal: Bytes::from(pps_nal.to_vec()),
        })
    }
}

/// Returns true iff the bytes of `nal` equal the bytes of `[hdr, ..data]`.
fn nal_matches(nal: &[u8], hdr: NalHeader, pieces: &[Bytes]) -> bool {
    let mut nal_bytes = Bytes::copy_from_slice(&nal[0..2]);
    if nal.is_empty() || nal_bytes.get_u16() != u16::from(hdr) {
        return false;
    }
    let mut nal_pos = 2;
    for piece in pieces {
        let new_pos = nal_pos + piece.len();
        if nal.len() < new_pos {
            return false;
        }
        if piece[..] != nal[nal_pos..new_pos] {
            return false;
        }
        nal_pos = new_pos;
    }
    nal_pos == nal.len()
}

/// Saves the given NAL to a contiguous Bytes.
fn to_bytes(hdr: NalHeader, len: u32, pieces: &[Bytes]) -> Bytes {
    let len = usize::try_from(len).expect("u32 fits in usize");
    let mut out = Vec::with_capacity(len);
    out.push((u16::from(hdr) >> 8) as u8);
    out.push((u16::from(hdr) & 0xFF) as u8);
    for piece in pieces {
        out.extend_from_slice(&piece[..]);
    }
    debug_assert_eq!(len, out.len());
    out.into()
}
