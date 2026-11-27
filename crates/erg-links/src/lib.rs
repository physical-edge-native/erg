//! erg-links: messages between the legs of the erg robot, in Dyne's wire format (Issue #33).
//!
//! Each leg sends its neighbours one message per control step with [`N_VALUES`] values: gravity direction (3),
//! angular velocity (3), foot contact (1), a learned vector (4) and its gait phase (2, cosine and sine; Issue #51).
//! The message is one Dyne `IFACE_BAND` message with one point of 13 quantities: a 24-byte header, the count and
//! per-point fields (8 bytes) and 104 bytes of values, 136 bytes in all. The encoding itself is Dyne's `dyne-wire`.
//!
//! The header fields are used as follows: `quantity` = inertial, `src` = the sending leg (0..4 in the order
//! front-left, front-right, hind-left, hind-right), `step` = the control step, `t_us` = the time stamp in
//! microseconds.
//!
//! The WebAssembly interface uses fixed buffers in linear memory, as Dyne's solvers do: the caller writes the
//! values to [`erg_values_buf`] and calls [`erg_encode`], which writes the message to [`erg_msg_buf`]; or it
//! writes a message there and calls [`erg_decode`], which writes the values back and keeps the header fields.


pub mod net;

use dyne_wire::{band_get, decode, encode, Header, Payload, KIND_IFACE_BAND, QTY_INERTIAL};

/// Values in one leg's message.
pub const N_VALUES: usize = 13;
/// Length of one leg's message in bytes: header 24 + count and per-point 8 + 13 values of 8 bytes.
pub const MSG_LEN: usize = dyne_wire::HDR_LEN + 8 + 8 * N_VALUES;
/// Capacity of the message buffer (room for longer messages, which decoding rejects).
pub const MSG_CAP: usize = 256;

/// Encode one leg's message into `buf`; returns the number of bytes written.
pub fn encode_leg(buf: &mut [u8], src: u32, step: u64, t_us: u64, values: &[f64; N_VALUES]) -> Result<usize, i32> {
    let mut data = [0u8; 8 * N_VALUES];
    for (i, v) in values.iter().enumerate() {
        data[8 * i..8 * i + 8].copy_from_slice(&v.to_bits().to_le_bytes());
    }
    let hdr = Header { kind: KIND_IFACE_BAND, quantity: QTY_INERTIAL, src, step, t_us };
    let payload = Payload::IfaceBand { data: &data, count: 1, per_point: N_VALUES as u32 };
    encode(buf, &hdr, &payload).map_err(|_| -1)
}

/// Decode one leg's message; returns the header and the values. Messages of another kind, quantity or shape are
/// rejected with a negative code.
pub fn decode_leg(buf: &[u8]) -> Result<(Header, [f64; N_VALUES]), i32> {
    let (hdr, payload, _) = decode(buf).map_err(|_| -1)?;
    if hdr.quantity != QTY_INERTIAL {
        return Err(-2);
    }
    match payload {
        Payload::IfaceBand { data, count: 1, per_point } if per_point as usize == N_VALUES => {
            let mut out = [0.0; N_VALUES];
            for (i, v) in out.iter_mut().enumerate() {
                *v = band_get(data, i);
            }
            Ok((hdr, out))
        }
        _ => Err(-3),
    }
}

// --- WebAssembly interface -------------------------------------------------------------------------------------------

static mut VALUES: [f64; N_VALUES] = [0.0; N_VALUES];
static mut MSG: [u8; MSG_CAP] = [0; MSG_CAP];
static mut LAST: Header = Header { kind: 0, quantity: 0, src: 0, step: 0, t_us: 0 };

/// Address of the values buffer (N_VALUES little-endian f64).
#[no_mangle]
pub extern "C" fn erg_values_buf() -> *mut f64 {
    core::ptr::addr_of_mut!(VALUES) as *mut f64
}

/// Address of the message buffer (MSG_CAP bytes).
#[no_mangle]
pub extern "C" fn erg_msg_buf() -> *mut u8 {
    core::ptr::addr_of_mut!(MSG) as *mut u8
}

/// Capacity of the message buffer in bytes.
#[no_mangle]
pub extern "C" fn erg_msg_cap() -> u32 {
    MSG_CAP as u32
}

/// Encode the values buffer into the message buffer; returns the message length, or a negative code.
#[no_mangle]
pub extern "C" fn erg_encode(src: u32, step: u64, t_us: u64) -> i32 {
    // SAFETY: the module is single-threaded and the buffers are only reached through these functions.
    let (values, msg) = unsafe { (&*core::ptr::addr_of!(VALUES), &mut *core::ptr::addr_of_mut!(MSG)) };
    match encode_leg(msg, src, step, t_us, values) {
        Ok(n) => n as i32,
        Err(e) => e,
    }
}

/// Decode the first `len` bytes of the message buffer into the values buffer; returns N_VALUES, or a negative code.
/// The header fields are then read with erg_last_src, erg_last_step and erg_last_t_us.
#[no_mangle]
pub extern "C" fn erg_decode(len: u32) -> i32 {
    // SAFETY: as in erg_encode.
    let (values, msg, last) = unsafe {
        (&mut *core::ptr::addr_of_mut!(VALUES), &*core::ptr::addr_of!(MSG), &mut *core::ptr::addr_of_mut!(LAST))
    };
    if len as usize > MSG_CAP {
        return -1;
    }
    match decode_leg(&msg[..len as usize]) {
        Ok((hdr, v)) => {
            *values = v;
            *last = hdr;
            N_VALUES as i32
        }
        Err(e) => e,
    }
}

/// Sending leg of the last decoded message.
#[no_mangle]
pub extern "C" fn erg_last_src() -> u32 {
    unsafe { (*core::ptr::addr_of!(LAST)).src }
}

/// Control step of the last decoded message.
#[no_mangle]
pub extern "C" fn erg_last_step() -> u64 {
    unsafe { (*core::ptr::addr_of!(LAST)).step }
}

/// Time stamp (microseconds) of the last decoded message.
#[no_mangle]
pub extern "C" fn erg_last_t_us() -> u64 {
    unsafe { (*core::ptr::addr_of!(LAST)).t_us }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_length() {
        let v = [1.0, -2.5, 0.0, -0.0, 1e-300, f64::INFINITY, 3.25, 0.5, -7.0, 2.0, 9.75, -1.0, 0.125];
        let mut buf = [0u8; MSG_CAP];
        let n = encode_leg(&mut buf, 2, 123, 456, &v).unwrap();
        assert_eq!(n, MSG_LEN);
        assert_eq!(MSG_LEN, 136);
        let (hdr, back) = decode_leg(&buf[..n]).unwrap();
        assert_eq!((hdr.src, hdr.step, hdr.t_us), (2, 123, 456));
        for (a, b) in v.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}

// --- Virtual network: WebAssembly interface (Issue #34) ---------------------------------------------------------------

static mut NET: Option<net::LegNet> = None;

/// Create (or replace) the network. `links`: 0 none, 1 neighbours, 2 all. Returns 0, or -1 for a bad setting.
#[no_mangle]
pub extern "C" fn erg_net_new(links: u32, loss: f64, lat_us: u64, jitter_us: u64, seed: u64, period_us: u64) -> i32 {
    let Some(links) = net::Links::from_u32(links) else { return -1 };
    if !(0.0..=1.0).contains(&loss) || period_us == 0 {
        return -1;
    }
    // SAFETY: single-threaded module.
    unsafe { *core::ptr::addr_of_mut!(NET) = Some(net::LegNet::new(links, loss, lat_us, jitter_us, seed, period_us)) };
    0
}

/// Bit mask of the legs that `leg` sends to (bit i = leg i), or 0 without a network.
#[no_mangle]
pub extern "C" fn erg_net_targets(leg: u32) -> u32 {
    let net = unsafe { (*core::ptr::addr_of!(NET)).as_ref() };
    match net {
        Some(n) if (leg as usize) < net::N_LEGS => n.links.targets(leg as usize).iter().map(|&t| 1u32 << t).sum(),
        _ => 0,
    }
}

/// Fate of a message from `from` to `to` sent at control step `step`: the delivery step, or -1 if lost, or -2
/// without a network.
#[no_mangle]
pub extern "C" fn erg_net_route(from: u32, to: u32, step: u64) -> i64 {
    let net = unsafe { (*core::ptr::addr_of_mut!(NET)).as_mut() };
    match net {
        Some(n) => match n.route(from as usize, to as usize, step) {
            Some(at) => at as i64,
            None => -1,
        },
        None => -2,
    }
}

/// Fates of one control step's messages, filled by [`erg_net_step`]: for each message, (source, destination,
/// delivery step or -1 if lost) as three i64.
static mut STEP_OUT: [i64; 3 * net::N_LEGS * (net::N_LEGS - 1)] = [0; 3 * net::N_LEGS * (net::N_LEGS - 1)];

/// Address of the buffer filled by [`erg_net_step`].
#[no_mangle]
pub extern "C" fn erg_net_step_buf() -> *mut i64 {
    core::ptr::addr_of_mut!(STEP_OUT) as *mut i64
}

/// Route the messages that every leg sends at control step `step`, in a fixed order (source leg 0..4, then its
/// destinations in increasing order) — the same order, and so the same random numbers, as calling
/// [`erg_net_route`] message by message. Returns the number of messages, or -2 without a network.
#[no_mangle]
pub extern "C" fn erg_net_step(step: u64) -> i32 {
    // SAFETY: single-threaded module.
    let (net, out) = unsafe { ((*core::ptr::addr_of_mut!(NET)).as_mut(), &mut *core::ptr::addr_of_mut!(STEP_OUT)) };
    let Some(n) = net else { return -2 };
    let mut k = 0;
    for src in 0..net::N_LEGS {
        for &dst in n.links.targets(src) {
            let at = n.route(src, dst, step).map_or(-1, |a| a as i64);
            out[3 * k..3 * k + 3].copy_from_slice(&[src as i64, dst as i64, at]);
            k += 1;
        }
    }
    k as i32
}
