const DEBUG: bool = false && cfg!(debug_assertions);
use field::{Field, PrimeField, FieldExtension};
pub trait FEXT: Field + FieldExtension<Self::F> {
    type F: PrimeField;
    fn slice(msg: &[Self]) -> &[field::baby_bear::ext4::BabyBearExt4];
    fn slice_to_u32(msg: &[Self]) -> &[u32];
    fn from_u32(msg: &[u32; Self::DEGREE]) -> Self;
    // VARIANT THAT KEEPS TYPES CLEANEST
    fn add_into(mut self, other: &Self) -> Self {
        *self.add_assign(other)
    }
    fn sub_into(mut self, other: &Self) -> Self {
        *self.sub_assign(other)
    }
    fn mul_into(mut self, other: &Self) -> Self {
        *self.mul_assign(other)
    }
    
    // VARIANT THAT REQUIRES LABELLING READS MUT EVERYWHERE
    // fn add_into(&mut self, other: &Self) -> Self {
    //     *self.add_assign(other)
    // }
    // fn sub_into(&mut self, other: &Self) -> Self {
    //     *self.sub_assign(other)
    // }
    // fn mul_into(&mut self, other: &Self) -> Self {
    //     *self.mul_assign(other)
    // }
}
impl FEXT for field::baby_bear::ext4::BabyBearExt4 {
    type F = field::baby_bear::base::BabyBearField;
    fn slice(msg: &[Self]) -> &[Self] { msg }
    fn slice_to_u32(msg: &[Self]) -> &[u32] {
        unsafe { core::slice::from_raw_parts(msg.as_ptr() as *const u32, 4*msg.len()) }
    }
    fn from_u32(msg: &[u32; 4]) -> Self {
        // dbg!(field::baby_bear::base::BabyBearField::from_u32_with_reduction(msg[0]).as_u32());
        Self {
            c0: field::baby_bear::ext2::BabyBearExt2 {
                c0: field::baby_bear::base::BabyBearField::from_u32_with_reduction(msg[0]),
                c1: field::baby_bear::base::BabyBearField::from_u32_with_reduction(msg[1]),
            },
            c1: field::baby_bear::ext2::BabyBearExt2 {
                c0: field::baby_bear::base::BabyBearField::from_u32_with_reduction(msg[2]),
                c1: field::baby_bear::base::BabyBearField::from_u32_with_reduction(msg[3]),
            },
        }
        // unsafe { core::slice::from_raw_parts(msg.as_ptr() as *const Self, 1) }
    }
}
fn from<E: FEXT>(x: u32) -> E {
    let f = E::F::from_u32(x).unwrap();
    E::from_base(f)
}
#[allow(non_snake_case)]
pub struct Transcript<E>
{
    IS_SEND: bool,
    hasher: blake2s_u32::DelegatedBlake2sState,
    buflen: usize,
    _keep: core::marker::PhantomData<E>
}
impl<E: FEXT> Transcript<E>
where [(); 8/E::DEGREE]:,
{
    const _CHECK: () = assert!(
        8 % E::DEGREE == 0 && 8 >= E::DEGREE
        && size_of::<[u32; 8]>() == size_of::<[E;  8 / E::DEGREE ]>()
        && align_of::<u32>() >= align_of::<E>()
    );
    pub fn from_seed(seed: [u32; 8]) -> Self {
        let seeded_hasher = {
            let mut hasher = blake2s_u32::DelegatedBlake2sState::new();
            hasher.state = seed; 
            hasher
        };
        Self {
            IS_SEND: false, // we start in squeeze mode
            hasher: seeded_hasher,
            buflen: 0, // WARN: if the seed is already used then set self.buflen to number of u32 values already consumed
            _keep: core::marker::PhantomData,
        }
    }
    pub fn to_seed(self) -> [u32; 8] {
        self.hasher.state
    }
    const NUM: usize = 0; // NUM OF STUFF THAT FITS...
    pub fn send(&mut self, msg: &[E]) {
        if self.IS_SEND == false {
            let seed = self.reset();
            self.part(&seed);
            self.IS_SEND = true;
        }
        self.part(Self::fext_to_u32(msg));
    }
    pub fn rand(&mut self) -> E {
        if self.IS_SEND == true {
            self.finalise();
            self.IS_SEND = false;
        }
        let already_used = self.buflen;
        let mut state = &self.hasher.state[already_used..];
        if already_used == 8 {
            let seed = self.reset();
            self.hasher.input_buffer[..8].copy_from_slice(&seed);
            self.hasher.input_buffer[8..].fill(0);
            self.buflen = 16; // fully filled up
            self.finalise();
            state = &self.hasher.state;
        }
        self.buflen += E::DEGREE;
        Self::u32_to_fext(&state[..E::DEGREE].try_into().unwrap())
    }

    fn reset(&mut self) -> [u32; 8] {
        let seed = self.hasher.state;
        self.hasher.reset();
        self.buflen = 0;
        seed
    }
    fn part(&mut self, mut msg: &[u32]) {
        let already_filled = self.buflen;
        let to_fill = 16 - already_filled;
        let total_input_len = already_filled + msg.len();

        let used = msg.len().min(to_fill);
        self.hasher.input_buffer[already_filled..][..used].copy_from_slice(&msg[..used]);
        self.buflen += used;
        msg = &msg[used..];

        // hash all but the last full block
        for _ in 1..total_input_len.div_ceil(16) {
            unsafe{self.hasher.run_round_function::<false>(16, false);}// full block, not reduced rounds, not final block
            let used = msg.len().min(16);
            self.hasher.input_buffer[..used].copy_from_slice(&msg[..used]);
            self.buflen = used;
            msg = &msg[used..];
        }
    }
    fn finalise(&mut self) {
        let already_filled = self.buflen;
        self.hasher.input_buffer[already_filled..].fill(0); // just a precaution
        unsafe{self.hasher.run_round_function::<false>(already_filled, true);}// final block, not reduced rounds
        self.buflen = 0; // just saying seed is full
    }
    pub fn fext_to_u32(msg: &[E]) -> &[u32] {
        // returns len = msg.len() * E::DEGREE
        let msg_u32 = unsafe{core::mem::transmute(msg)};
        // let info = format!("E:{msg:?} -> u32:{msg_u32:?}"); dbg!(info);
        msg_u32
    }
    pub fn u32_to_fext(msg: &[u32; E::DEGREE]) -> E {
        assert!(size_of::<E>() == E::DEGREE * size_of::<u32>());
        let msg_fext = E::from_u32(msg);
        // let msg_fext = unsafe { core::ptr::read_unaligned(msg.as_ptr() as *const E) };
        // let msg_fext = unsafe { core::ptr::read_unaligned(msg.as_ptr().cast::<E>()) };
        // let msg_fext = unsafe{core::mem::transmute(msg)};
        // let info = format!("u32:{msg:?} -> E:{msg_fext:?}"); dbg!(info);
        msg_fext
    }
}


// sumcheck1: sum_x eq() * A(x) * B(x)
// sumcheck2: sum_x eq() * (A(x) + C(x)*B(x)*4 )...
// ...
// --> make 1 batched sumcheck
//
// L: plaintext 32 values, L(0), L(1), ..
// select random r
// L(r) == sum_{0,1}^5 eq(x1..x5, r1..r5) * L(x)
//        == sum_{0,1}^5 eq(x1..x5, r1..r5) * L'(0, x1..x5) * L'(1, x1..x5)
// L'(0, rr1..rr5) * L'(1, rr1..rr5)
// --> L'(rrr1..rrr6) at point (rrr1..rrr6)
// L'(rrr1..rrr6) == sum_{0,1}^6 eq(x1..x6, rrr1..rrr6) * L'(x1..x6) where you swap out L' with L''
// L'''''''(x) shrinking layer
// C(R1..R24) == A(x)*B(x)
// WHIR=FRI+SUMCHECK
//
fn gen_eqtable_linear<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let one_minus_z = z.map(|zi| E::ONE.sub_into(&zi));
    let m = z.len();
    (0..1<<m).map(|mut i| {
        let mut el = E::ONE;
        for (zi, one_minus_zi) in z.iter().zip(&one_minus_z) {
            if i & 1 == 1 {
                el.mul_assign(zi);
            } else {
                el.mul_assign(one_minus_zi);
            }
            i >>= 1;
        }
        el
    }).collect::<Vec<_>>()
}
fn gen_eqtable_recursive_lowtohigh<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M);
    unsafe{ out.set_len(1<<M); }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z {
        // let one_minus_zi = E::ONE.sub_into(zi);
        for i in 0..len {
            let old = out[i];
            let new_secondhalf = old.clone().mul_into(zi);
            let new = old.sub_into(&new_secondhalf);
            // let new = old.mul_into(&one_minus_zi);
            out[i] = new;
            out[i + len] = new_secondhalf;
        }
        len <<= 1;
    }
    out
}
fn gen_eqtable_recursive_lowtohigh_antiscatter<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M);
    let mut buf = Vec::with_capacity(1<<M);
    unsafe{
        out.set_len(1<<M);
        buf.set_len(1<<M);
    }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[i];
            let new_secondhalf = old.clone().mul_into(zi);
            let new = old.sub_into(&new_secondhalf); // old - old*zi = old*(1-zi)
            // let new = old * one_minus_zi % P; // alternatively
            buf[i] = new;
            buf[i + len] = new_secondhalf;
        }
        core::mem::swap(&mut out, &mut buf);
        len <<= 1;
    }
    out
}
fn gen_eqtable_recursive_hightolow_slow<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M);
    unsafe{ out.set_len(1<<M); }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z.into_iter().rev() {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in (0..len).rev() {
            let old = out[i];
            let new_secondhalf_odd = old.clone().mul_into(zi);
            let new_secondhalf_even = old.sub_into(&new_secondhalf_odd); // old - old*zi = old*(1-zi)
            // let new_secondhalf_even = old * one_minus_zi % P; // alternatively
            out[i*2] = new_secondhalf_even;
            out[i*2 + 1] = new_secondhalf_odd;
        }
        len <<= 1;
    }
    out
}
fn gen_eqtable_recursive_hightolow_antiscatter<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M);
    let mut buf = Vec::with_capacity(1<<M); // anti-scatter ping-pong buffer
    unsafe{
        out.set_len(1<<M);
        buf.set_len(1<<M);
    }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z.into_iter().rev() {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[i];
            let new_secondhalf_odd = old.clone().mul_into(zi);
            let new_secondhalf_even = old.sub_into(&new_secondhalf_odd); // old - old*zi = old*(1-zi)
            // let new_secondhalf_even = old * one_minus_zi % P; // alternatively
            buf[i*2] = new_secondhalf_even;
            buf[i*2 + 1] = new_secondhalf_odd;
        }
        core::mem::swap(&mut out, &mut buf);
        len <<= 1;
    }
    out
}
fn gen_multieqtable_recursive_lowtohigh<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M+1);
    unsafe{ out.set_len(1<<M+1); }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[len - 1..][i];
            let new_secondhalf = old.clone().mul_into(zi);
            let new = old.sub_into(&new_secondhalf); // old - old*zi = old*(1-zi)
            // let new = old * one_minus_zi % P; // alternatively
            out[len*2 - 1..][i] = new;
            out[len*2 - 1..][i + len] = new_secondhalf;
        }
        len <<= 1;
    }
    out
}
fn gen_multiaccsumtable_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(trace1: &[E], trace2: &[E]) -> (Vec<E>, Vec<E>) {
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let mut num = Vec::with_capacity(1<<M);
    let mut den = Vec::with_capacity(1<<M);
    unsafe{
        num.set_len(1<<M);
        den.set_len(1<<M);
    }
    let mut len = 1<<M-1;
    for li in (SKIPLAYERS..M).rev() {
        for i in 0..len {
            let (old_num_even, old_num_odd) = if li==M-1 {
                (trace1[i*2], trace1[i*2+1])
            } else {
                (num[len*2 - 1..][i*2], num[len*2 - 1..][i*2+1])
            };
            let (old_den_even, old_den_odd) = if li==M-1 {
                (trace2[i*2], trace2[i*2+1])
            } else {
                (den[len*2 - 1..][i*2], den[len*2 - 1..][i*2+1])
            };
            let new_num = old_num_even.mul_into(&old_den_odd).add_into(&old_num_odd.mul_into(&old_den_even));
            let new_den = old_den_even.mul_into(&old_den_odd);
            num[len - 1..][i] = new_num;
            den[len - 1..][i] = new_den;
        }
        len >>= 1;
    }
    (num, den)
}
fn gen_multiaccsumtable_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(trace1: &[E], trace2: &[E]) -> (Vec<E>, Vec<E>) {
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let mut num = Vec::with_capacity(1<<M);
    let mut den = Vec::with_capacity(1<<M);
    unsafe{
        num.set_len(1<<M);
        den.set_len(1<<M);
    }
    let mut len = 1<<M-1;
    for li in (SKIPLAYERS..M).rev() {
        for i in 0..len {
            let (old_num_low, old_num_high) = if li==M-1 {
                (trace1[i], trace1[i+len])
            } else {
                (num[len*2 - 1..][i], num[len*2 - 1..][i+len])
            };
            let (old_den_low, old_den_high) = if li==M-1 {
                (trace2[i], trace2[i+len])
            } else {
                (den[len*2 - 1..][i], den[len*2 - 1..][i+len])
            };
            let new_num = old_num_low.mul_into(&old_den_high).add_into(&old_num_high.mul_into(&old_den_low));
            let new_den = old_den_low.mul_into(&old_den_high);
            num[len - 1..][i] = new_num;
            den[len - 1..][i] = new_den;
        }
        len >>= 1;
    }
    (num, den)
}
fn gen_multiaccprodtable_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(trace: &[E]) -> Vec<E> {
    assert!(trace.len() == 1<<M);
    let mut out = Vec::with_capacity(1<<M);
    unsafe{ out.set_len(1<<M); }
    let mut len = 1<<M-1;
    for li in (SKIPLAYERS..M).rev() {
        for i in 0..len {
            let (old_even, old_odd) = if li==M-1 {
                (trace[i*2], trace[i*2+1])
            } else {
                (out[len*2 - 1..][i*2], out[len*2 - 1..][i*2+1])
            };
            let new = old_even.mul_into(&old_odd);
            out[len - 1..][i] = new;
        }
        len >>= 1;
    }
    out
}
fn gen_multiaccprodtable_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(trace: &[E]) -> Vec<E> {
    assert!(trace.len() == 1<<M);
    let mut out = Vec::with_capacity(1<<M);
    unsafe{ out.set_len(1<<M); }
    let mut len = 1<<M-1;
    for li in (SKIPLAYERS..M).rev() {
        for i in 0..len {
            let (old_low, old_high) = if li==M-1 {
                (trace[i], trace[i+len])
            } else {
                (out[len*2 - 1..][i], out[len*2 - 1..][i+len])
            };
            let new = old_low.mul_into(&old_high);
            out[len - 1..][i] = new;
        }
        len >>= 1;
    }
    out
}
fn gen_multieqtable_recursive_lowtohigh_antiscatter<const M: usize, E: FEXT>(z: &[E; M]) -> (Vec<E>, Vec<E>) {
    let mut out = Vec::with_capacity(1<<M+1);
    let mut buf = Vec::with_capacity(1<<M+1);
    unsafe{
        out.set_len(1<<M+1);
        buf.set_len(1<<M+1);
    }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[len - 1..][i];
            let new_secondhalf = old.clone().mul_into(zi);
            let new = old.sub_into(&new_secondhalf); // old - old*zi = old*(1-zi)
            // let new = old * one_minus_zi % P; // alternatively
            buf[len*2 - 1..][i] = new;
            buf[len*2 - 1..][i + len] = new_secondhalf;
        }
        core::mem::swap(&mut out, &mut buf);
        len <<= 1;
    }
    (out, buf)
}
fn gen_multieqtable_recursive_hightolow_antiscatter<const M: usize, E: FEXT>(z: &[E; M]) -> (Vec<E>, Vec<E>) {
    let mut out = Vec::with_capacity(1<<M+1);
    let mut buf = Vec::with_capacity(1<<M+1);
    unsafe{
        out.set_len(1<<M+1);
        buf.set_len(1<<M+1);
    }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z.into_iter().rev() {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[len - 1..][i];
            let new_secondhalf_odd = old.clone().mul_into(zi);
            let new_secondhalf_even = old.sub_into(&new_secondhalf_odd); // old - old*zi = old*(1-zi)
            // let new_secondhalf_even = old * one_minus_zi % P; // alternatively
            buf[len*2 - 1..][i*2] = new_secondhalf_even;
            buf[len*2 - 1..][i*2 + 1] = new_secondhalf_odd;
        }
        core::mem::swap(&mut out, &mut buf);
        len <<= 1;
    }
    (out, buf)
}
fn gen_multieqtable_recursive_hightolow<const M: usize, E: FEXT>(z: &[E; M]) -> Vec<E> {
    let mut out = Vec::with_capacity(1<<M+1);
    unsafe{ out.set_len(1<<M+1); }
    out[0] = E::ONE;
    let mut len = 1;
    for zi in z.into_iter().rev() {
        // let one_minus_zi = (P - zi + 1) % P;
        for i in 0..len {
            let old = out[len - 1..][i];
            let new_secondhalf_odd = old.clone().mul_into(zi);
            let new_secondhalf_even = old.sub_into(&new_secondhalf_odd); // old - old*zi = old*(1-zi)
            // let new_secondhalf_even = old * one_minus_zi % P; // alternatively
            out[len*2 - 1..][i*2] = new_secondhalf_even;
            out[len*2 - 1..][i*2 + 1] = new_secondhalf_odd;
        }
        len <<= 1;
    }
    out
}
fn slice_multieqtable<const M: usize, E: FEXT>(multieq: &[E]) -> [&[E]; M+1] {
    assert!(multieq.len() == 1<<M+1);
    core::array::from_fn(|i|
        &multieq[(1<<i) - 1..(1<<i+1) - 1] // eqs
    )
}
fn slicemut_multiacctable<'a, 'b, const M: usize, E: FEXT>(multiacc: &'b mut [E], trace:&'a mut [E]) -> [&'b mut[E]; M+1] where 'a:'b {
    assert!(multiacc.len() == 1<<M && trace.len() == 1<<M);
    core::array::from_fn(|i|
        if i == M {
            // &trace[..]
            unsafe{ core::slice::from_raw_parts_mut(trace.as_mut_ptr(), 1<<i) }
        } else {
            // &multiacc[(1<<i) - 1..(1<<i+1) - 1] // eqs
            unsafe{ core::slice::from_raw_parts_mut(multiacc.as_mut_ptr().add((1<<i)-1), 1<<i) }
        }
    )
}
fn slicerev_multieqtable<const M: usize, E: FEXT>(multieq: &[E]) -> [&[E]; M+1] {
    assert!(multieq.len() == 1<<M+1);
    core::array::from_fn(|i|
        &multieq[(1<<M-i) - 1..(1<<M+1-i) - 1] // eqs
    )
}
fn slice_multieqtable_antiscattered<const M: usize, E: FEXT>(multieq_scattered: &(Vec<E>, Vec<E>)) -> [&[E]; M+1] {
    let (out, buf) = multieq_scattered;
    assert!(out.len() == 1<<M+1 && buf.len() == 1<<M+1);
    core::array::from_fn(|i| // &eqs
        if (i % 2 == 0) ^ (M % 2 == 0) {
            &buf[(1<<i) - 1..(1<<i+1) - 1]

        } else {
            &out[(1<<i) - 1..(1<<i+1) - 1]
        }
    )
}
fn slicerev_multieqtable_antiscattered<const M: usize, E: FEXT>(multieq_scattered: &(Vec<E>, Vec<E>)) -> [&[E]; M+1] {
    let (out, buf) = multieq_scattered;
    assert!(out.len() == 1<<M+1 && buf.len() == 1<<M+1);
    core::array::from_fn(|i| // &eqs
        if ((M-i+1) % 2 == 0) ^ (M % 2 == 0) {
            &out[(1<<M-i) - 1..(1<<M+1-i) - 1]

        } else {
            &buf[(1<<M-i) - 1..(1<<M+1-i) - 1]
        }
    )
}
fn randeval_lowtohigh<const M: usize, E: FEXT>(trace: &mut [E], z: &[E; M]) -> E {
    assert!(trace.len() == 1<<M);
    // repeatedly fold
    if DEBUG {
        // let layer = format!("{trace:?}");
        // dbg!(layer);
    }
    for round in 0..M {
        let step = 1<<round;
        let r = z[round];
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            let (even, odd) = (trace[ie], trace[io]);
            let sub = odd.sub_into(&even);
            let fold = even.add_into(&r.mul_into(&sub));
            trace[ie] = fold;
        }
        if DEBUG {
            // let poly_fold = trace.iter_mut().step_by(1<<round+1).collect::<Vec<&mut u8>>();
            // dbg!(poly_fold);
        }
    }
    trace[0]
}
fn randeval_hightolow<const M: usize, E: FEXT>(trace: &mut [E], z: &[E; M]) -> E {
    assert!(trace.len() == 1<<M);
    // repeatedly fold
    for round in 0..M {
        let step = 1<<M-round-1;
        let r = z[M-1-round];
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            let (even, odd) = (trace[ie], trace[io]);
            let sub = odd.sub_into(&even);
            let fold = even.add_into(&r.mul_into(&sub));
            trace[ie] = fold;
        }
    }
    trace[0]
}
fn randeval_pairmult_lowtohigh<const M: usize, E: FEXT>(trace1: &mut [E], trace2: &[E], z: &[E; M]) -> E {
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    // repeatedly fold
    if DEBUG {
        // dbg!(&trace1, &trace2);
    }
    for round in 0..1 {
        let step = 1<<round;
        let r = z[round];
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            let (even1, odd1) = (trace1[ie], trace1[io]);
            let (even2, odd2) = (trace2[ie], trace2[io]);
            let even = even1.mul_into(&even2);
            let odd = odd1.mul_into(&odd2);
            let sub = odd.sub_into(&even);
            let fold = even.add_into(&r.mul_into(&sub));
            trace1[ie] = fold;
        }
        if DEBUG {
            // let poly_fold = trace1.iter_mut().step_by(1<<round+1).collect::<Vec<&mut u8>>();
            // dbg!(poly_fold);
        }
    }
    for round in 1..M {
        let step = 1<<round;
        let r = z[round];
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            let (even, odd) = (trace1[ie], trace1[io]);
            let sub = odd.sub_into(&even);
            let fold = even.add_into(&r.mul_into(&sub));
            trace1[ie] = fold;
        }
        if DEBUG {
            // let poly_fold = trace1.iter_mut().step_by(1<<round+1).collect::<Vec<&mut u8>>();
            // dbg!(poly_fold);
        }
    }
    trace1[0]
}
fn randeval_pairmult_hightolow<const M: usize, E: FEXT>(trace1: &mut [E], trace2: &[E], z: &[E; M]) -> E {
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    // repeatedly fold
    for round in 0..1 {
        let step = 1<<M-round-1;
        let r = z[M-1-round];
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            let (low1,  high1) = (trace1[ie], trace1[io]);
            let (low2,  high2) = (trace2[ie], trace2[io]);
            let low = low1.mul_into(&low2);
            let high = high1.mul_into(&high2);
            let sub = high.sub_into(&low);
            let fold = low.add_into(&r.mul_into(&sub));
            trace1[ie] = fold;
        }
    }
    for round in 1..M {
        let step = 1<<M-round-1;
        let r = z[M-1-round];
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            let (low, high) = (trace1[ie], trace1[io]);
            let sub = high.sub_into(&low);
            let fold = low.add_into(&r.mul_into(&sub));
            trace1[ie] = fold;
        }
    }
    trace1[0]
}





#[cfg(test)]
mod tests {
    use super::*;
    #[allow(non_snake_case)]
    #[test]
    fn test_all_ops() {
        type F = field::baby_bear::base::BabyBearField;
        type E = field::baby_bear::ext4::BabyBearExt4;
        const M: usize = 10; // CHANGE AT WILL FOR TESTING

        let seed: [u32; 8] = [12312312,3453453,0640596904,45605943,232302349,2301230491,1340343902,2340234023];
        let mut tx: Transcript<E> = Transcript::from_seed(seed);
        let z: [E; M] = core::array::from_fn(|_| tx.rand());
        dbg!(z);

        ///////// SINGLEEQ
        let mut t_gen_eqtable = vec![];

        let time = std::time::Instant::now();
        let eq1 = gen_eqtable_recursive_lowtohigh(&z);
        t_gen_eqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let eq2 = gen_eqtable_recursive_lowtohigh_antiscatter(&z);
        t_gen_eqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let eq3 = gen_eqtable_recursive_hightolow_slow(&z);
        t_gen_eqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let eq4 = gen_eqtable_recursive_hightolow_antiscatter(&z);
        t_gen_eqtable.push(time.elapsed());

        assert!(eq1 == eq2);
        assert!(eq1 == eq3);
        assert!(eq1 == eq4);
        dbg!(t_gen_eqtable);
        /////////////////////// MULTIEQ
        let mut t_gen_multieqtable = vec![];

        let time = std::time::Instant::now();
        let multieq_lowtohigh = gen_multieqtable_recursive_lowtohigh(&z);
        t_gen_multieqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let multieq_lowtohigh_scattered = gen_multieqtable_recursive_lowtohigh_antiscatter(&z);
        t_gen_multieqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let multieq_hightolow = gen_multieqtable_recursive_hightolow(&z);
        t_gen_multieqtable.push(time.elapsed());

        let time = std::time::Instant::now();
        let multieq_hightolow_scattered = gen_multieqtable_recursive_hightolow_antiscatter(&z);
        t_gen_multieqtable.push(time.elapsed());

        let eqs_lowtohigh = slice_multieqtable::<M, _>(&multieq_lowtohigh);
        let eqs_lowtohigh_2 = slice_multieqtable_antiscattered::<M, _>(&multieq_lowtohigh_scattered);
        let eqs_hightolow = slice_multieqtable::<M, _>(&multieq_hightolow);
        let eqs_hightolow_2 = slice_multieqtable_antiscattered::<M, _>(&multieq_hightolow_scattered);
        seq_macro::seq!(i in 0..50 { if (0..=M).contains(&i) { // M+1 rounds
            let multiz_low: &[E; i] = z.split_array_ref::<i>().0;
            let multiz_high: &[E; i] = z.rsplit_array_ref::<i>().1;
            let eq_low = &gen_eqtable_recursive_lowtohigh(&multiz_low);
            let eq_high = &gen_eqtable_recursive_lowtohigh(&multiz_high);
            assert!(eq_low == eqs_lowtohigh.get(i).unwrap());
            assert!(eq_low == eqs_lowtohigh_2.get(i).unwrap());
            assert!(eq_high == eqs_hightolow.get(i).unwrap());
            assert!(eq_high == eqs_hightolow_2.get(i).unwrap());
        }});
        let eqs_lowtohigh_rev = slicerev_multieqtable::<M, _>(&multieq_lowtohigh);
        let eqs_lowtohigh_rev_2 = slicerev_multieqtable_antiscattered::<M, _>(&multieq_lowtohigh_scattered);
        let eqs_hightolow_rev = slicerev_multieqtable::<M, _>(&multieq_hightolow);
        let eqs_hightolow_rev_2 = slicerev_multieqtable_antiscattered::<M, _>(&multieq_hightolow_scattered);
        seq_macro::seq!(i in 0..50 { if (0..=M).contains(&i) { // M+1 rounds
            #[allow(unused_comparisons)]
            const M_MINUS_I: usize = if i<= M {M-i} else {0};
            let multiz_low_rev: &[E; M_MINUS_I] = z.split_array_ref::<M_MINUS_I>().0;
            let multiz_high_rev: &[E; M_MINUS_I] = z.rsplit_array_ref::<M_MINUS_I>().1;
            let eq_low_rev = &gen_eqtable_recursive_lowtohigh(&multiz_low_rev);
            let eq_high_rev = &gen_eqtable_recursive_lowtohigh(&multiz_high_rev);
            assert!(eqs_lowtohigh_rev.get(i).unwrap() == eq_low_rev);
            assert!(eqs_lowtohigh_rev_2.get(i).unwrap() == eq_low_rev);
            assert!(eqs_hightolow_rev.get(i).unwrap() == eq_high_rev);
            assert!(eqs_hightolow_rev_2.get(i).unwrap() == eq_high_rev);
        } });
        dbg!(t_gen_multieqtable);
        ////////////////// SUMCHECK
        let trace1 = (0..1<<M).map(|_| tx.rand()).collect::<Vec<_>>();
        let trace2 = (0..1<<M).map(|_| tx.rand()).collect::<Vec<_>>();

        let mut t_randeval = vec![];

        let mut trace1_clone = trace1.clone();
        let time = std::time::Instant::now();
        let claim1 = randeval_lowtohigh(&mut trace1_clone, &z);
        t_randeval.push(time.elapsed());

        let mut trace1_clone = trace1.clone();
        let time = std::time::Instant::now();
        let claim1_2 = randeval_hightolow(&mut trace1_clone, &z);
        t_randeval.push(time.elapsed());

        let mut trace1_clone = trace1.clone();
        let time = std::time::Instant::now();
        let claim3 = randeval_pairmult_lowtohigh(&mut trace1_clone, &trace2, &z);
        t_randeval.push(time.elapsed());

        let mut trace1_clone = trace1.clone();
        let time = std::time::Instant::now();
        let claim3_2 = randeval_pairmult_hightolow(&mut trace1_clone, &trace2, &z);
        t_randeval.push(time.elapsed());

        assert!(claim1 == claim1_2);
        assert!(claim3 == claim3_2);
        dbg!(t_randeval);

        #[allow(non_snake_case)]
        let mut t_sum_eqspliteqAB = vec![];
        // let mut pi_sum_eqspliteqAB = vec![];

        let mut trace1_clone = trace1.clone();
        let mut trace2_clone = trace2.clone();
        let mut z_clone = z.clone();
        let time = std::time::Instant::now();
        // reset();
        sum_eqspliteqAB_lowtohigh::<_, _, 0>(&mut tx, &mut trace1_clone, &mut trace2_clone, &mut z_clone, claim3);
        t_sum_eqspliteqAB.push(time.elapsed());
        // pi_sum_eqspliteqAB.push(unsafe{TRANSCRIPT.len()});

        let mut trace1_clone = trace1.clone();
        let mut trace2_clone = trace2.clone();
        let mut z_clone = z.clone();
        // reset();
        let time = std::time::Instant::now();
        sum_eqspliteqAB_hightolow::<_, _, 0>(&mut tx, &mut trace1_clone, &mut trace2_clone, &mut z_clone, claim3);
        t_sum_eqspliteqAB.push(time.elapsed());
        // pi_sum_eqspliteqAB.push(unsafe{TRANSCRIPT.len()});

        dbg!(t_sum_eqspliteqAB);
        // dbg!(pi_sum_eqspliteqAB);
        ////////////////// GKR
        let trace1 = {
            let mut trace = (0..1<<M).map(|_| {let r = tx.rand(); if r==E::ZERO {E::ONE} else {r}}).collect::<Vec<_>>();
            let acc = trace[..trace.len()-1].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
            trace[(1<<M)-1] = acc.inverse().unwrap(); // simulating valid occurrences
            trace
        };
        let (trace2, trace3) = {
            const SMALL: usize = 3;
            let mut numerator = (0..1<<M).map(|_| tx.rand()).collect::<Vec<_>>();
            let mut denominator = (0..1<<M).map(|_| {
                let f: F = tx.rand().into_coeffs_in_base()[0];
                E::from_base(F::from_u32_with_reduction((f.as_u32_reduced() % SMALL as u32)+1)) 
            }).collect::<Vec<_>>();
            for i in 0..SMALL {
                numerator[(1<<M)-SMALL+i] = E::ZERO;
                denominator[(1<<M)-SMALL+i] = from(i as u32 + 1);
            }
            for i in 0..(1<<M)-SMALL {
                let (value, table_f): (E, F) = (numerator[i], denominator[i].into_coeffs_in_base()[0]);
                let table = table_f.as_u32_reduced() as usize;
                numerator[(1<<M)-SMALL+table-1] = numerator[(1<<M)-SMALL+table-1].sub_into(&value);
            }
            let safe = numerator.iter().zip(&denominator).fold([E::ZERO; SMALL+1], |mut acc, (val, tab_e)| {
                let tab_f: F = tab_e.into_coeffs_in_base()[0];
                let tab = tab_f.as_u32_reduced() as usize;
                acc[tab] = acc[tab].add_into(&val);
                acc
            });
            assert!(safe == [E::ZERO; SMALL+1]);
            (numerator, denominator)
        };
        let (trace4, trace5) = {
            const SMALL: usize = 3;
            let mut numerator = (0..1<<M).map(|_| tx.rand()).collect::<Vec<_>>();
            let mut denominator = (0..1<<M).map(|_| {
                let f: F = tx.rand().into_coeffs_in_base()[0];
                E::from_base(F::from_u32_with_reduction((f.as_u32_reduced() % SMALL as u32)+1)) 
            }).collect::<Vec<_>>();
            for i in 0..SMALL {
                numerator[(1<<M)-SMALL+i] = E::ZERO;
                denominator[(1<<M)-SMALL+i] = from(i as u32 + 1);
            }
            for i in 0..(1<<M)-SMALL {
                let (value, table_f): (E, F) = (numerator[i], denominator[i].into_coeffs_in_base()[0]);
                let table = table_f.as_u32_reduced() as usize;
                numerator[(1<<M)-SMALL+table-1] = numerator[(1<<M)-SMALL+table-1].sub_into(&value);
            }
            let safe = numerator.iter().zip(&denominator).fold([E::ZERO; SMALL+1], |mut acc, (val, tab_e)| {
                let tab_f: F = tab_e.into_coeffs_in_base()[0];
                let tab = tab_f.as_u32_reduced() as usize;
                acc[tab] = acc[tab].add_into(&val);
                acc
            });
            assert!(safe == [E::ZERO; SMALL+1]);
            (numerator, denominator)
        };
        let (trace6, trace7) = {
            const SMALL: usize = 3;
            let mut numerator = (0..1<<M).map(|_| tx.rand()).collect::<Vec<_>>();
            let mut denominator = (0..1<<M).map(|_| {
                let f: F = tx.rand().into_coeffs_in_base()[0];
                E::from_base(F::from_u32_with_reduction((f.as_u32_reduced() % SMALL as u32)+1)) 
            }).collect::<Vec<_>>();
            for i in 0..SMALL {
                numerator[(1<<M)-SMALL+i] = E::ZERO;
                denominator[(1<<M)-SMALL+i] = from(i as u32 + 1);
            }
            for i in 0..(1<<M)-SMALL {
                let (value, table_f): (E, F) = (numerator[i], denominator[i].into_coeffs_in_base()[0]);
                let table = table_f.as_u32_reduced() as usize;
                numerator[(1<<M)-SMALL+table-1] = numerator[(1<<M)-SMALL+table-1].sub_into(&value);
            }
            let safe = numerator.iter().zip(&denominator).fold([E::ZERO; SMALL+1], |mut acc, (val, tab_e)| {
                let tab_f: F = tab_e.into_coeffs_in_base()[0];
                let tab = tab_f.as_u32_reduced() as usize;
                acc[tab] = acc[tab].add_into(&val);
                acc
            });
            assert!(safe == [E::ZERO; SMALL+1]);
            (numerator, denominator)
        };

        let mut t_gkr_grandproduct = vec![];
        // let mut pi_gkr_grandproduct = vec![];

        let mut trace1_clone = trace1.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_grandproduct_lowtohigh::<M, _, 2>(&mut tx, &mut trace1_clone);
        t_gkr_grandproduct.push(time.elapsed());
        // pi_gkr_grandproduct.push(unsafe{TRANSCRIPT.len()});

        let mut trace1_clone = trace1.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_grandproduct_hightolow::<M, _, 2>(&mut tx, &mut trace1_clone);
        t_gkr_grandproduct.push(time.elapsed());
        // pi_gkr_grandproduct.push(unsafe{TRANSCRIPT.len()});

        dbg!(t_gkr_grandproduct);
        // dbg!(pi_gkr_grandproduct);

        let mut t_gkr_grandsum = vec![];
        // let mut pi_gkr_grandsum = vec![];

        let mut trace2_clone = trace2.clone();
        let mut trace3_clone = trace3.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_grandsum_lowtohigh::<M, _, 5>(&mut tx, &mut trace2_clone, &mut trace3_clone);
        t_gkr_grandsum.push(time.elapsed());
        // pi_gkr_grandsum.push(unsafe{TRANSCRIPT.len()});

        let mut trace2_clone = trace2.clone();
        let mut trace3_clone = trace3.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_grandsum_hightolow::<M, _, 5>(&mut tx, &mut trace2_clone, &mut trace3_clone);
        t_gkr_grandsum.push(time.elapsed());
        // pi_gkr_grandsum.push(unsafe{TRANSCRIPT.len()});

        dbg!(t_gkr_grandsum);
        // dbg!(pi_gkr_grandsum);
        let mut t_gkr_kernelsum = vec![];
        // let mut pi_gkr_kernelsum = vec![];

        let mut trace2_clone = trace2.clone();
        let mut trace3_clone = trace3.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_kernelgrandsum_lowtohigh::<M, _, 5>(&mut tx, &mut trace2_clone, &mut trace3_clone);
        t_gkr_kernelsum.push(time.elapsed());
        // pi_gkr_kernelsum.push(unsafe{TRANSCRIPT.len()});

        let mut trace2_clone = trace2.clone();
        let mut trace3_clone = trace3.clone();
        // reset();
        let time = std::time::Instant::now();
        gkr_kernelgrandsum_hightolow::<M, _, 5>(&mut tx, &mut trace2_clone, &mut trace3_clone);
        t_gkr_kernelsum.push(time.elapsed());
        // pi_gkr_kernelsum.push(unsafe{TRANSCRIPT.len()});

        dbg!(t_gkr_kernelsum);
        // dbg!(pi_gkr_kernelsum);
        let mut t_gkr_kernelfull = vec![];
        // let mut pi_gkr_kernelfull = vec![];

        let mut trace_readset = trace1.clone();
        let mut trace_writeset = trace1.iter().rev().copied().collect::<Vec<E>>();
        let (mut trace_lookupnum, mut trace_lookupden) = (trace2.clone(), trace3.clone());
        let (mut trace_rangechecknum, mut trace_rangecheckden) = (trace4.clone(), trace5.clone());
        let (mut trace_timechecknum, mut trace_timecheckden) = (trace6.clone(), trace7.clone());
        // reset();
        let time = std::time::Instant::now();
        gkr_kernelfull_lowtohigh::<M, _, 5>(&mut tx, &mut trace_readset, &mut trace_writeset, &mut trace_lookupnum, &mut trace_lookupden, &mut trace_rangechecknum, &mut trace_rangecheckden, &mut trace_timechecknum, &mut trace_timecheckden);
        t_gkr_kernelfull.push(time.elapsed());
        // pi_gkr_kernelfull.push(unsafe{TRANSCRIPT.len()});

        let mut trace_readset = trace1.clone();
        let mut trace_writeset = trace1.iter().rev().copied().collect::<Vec<E>>();
        let (mut trace_lookupnum, mut trace_lookupden) = (trace2.clone(), trace3.clone());
        let (mut trace_rangechecknum, mut trace_rangecheckden) = (trace4.clone(), trace5.clone());
        let (mut trace_timechecknum, mut trace_timecheckden) = (trace6.clone(), trace7.clone());
        // reset();
        let time = std::time::Instant::now();
        gkr_kernelfull_hightolow::<M, _, 5>(&mut tx, &mut trace_readset, &mut trace_writeset, &mut trace_lookupnum, &mut trace_lookupden, &mut trace_rangechecknum, &mut trace_rangecheckden, &mut trace_timechecknum, &mut trace_timecheckden);
        t_gkr_kernelfull.push(time.elapsed());
        // pi_gkr_kernelfull.push(unsafe{TRANSCRIPT.len()});

        dbg!(&t_gkr_kernelfull);
        // dbg!(pi_gkr_kernelfull);
        let mut t_gkr_full = vec![];
        // let mut pi_gkr_full = vec![];

        let mut trace_readset = trace1.clone();
        let mut trace_writeset = trace1.iter().rev().copied().collect::<Vec<E>>();
        let (mut trace_lookupnum, mut trace_lookupden) = (trace2.clone(), trace3.clone());
        let (mut trace_rangechecknum, mut trace_rangecheckden) = (trace4.clone(), trace5.clone());
        let (mut trace_timechecknum, mut trace_timecheckden) = (trace6.clone(), trace7.clone());
        // reset();
        let time = std::time::Instant::now();
        gkr_full_lowtohigh::<M, _, 5>(&mut tx, &mut trace_readset, &mut trace_writeset, &mut trace_lookupnum, &mut trace_lookupden, &mut trace_rangechecknum, &mut trace_rangecheckden, &mut trace_timechecknum, &mut trace_timecheckden);
        t_gkr_full.push(time.elapsed());
        // pi_gkr_full.push(unsafe{TRANSCRIPT.len()});


        let mut trace_readset = trace1.clone();
        let mut trace_writeset = trace1.iter().rev().copied().collect::<Vec<E>>();
        let (mut trace_lookupnum, mut trace_lookupden) = (trace2.clone(), trace3.clone());
        let (mut trace_rangechecknum, mut trace_rangecheckden) = (trace4.clone(), trace5.clone());
        let (mut trace_timechecknum, mut trace_timecheckden) = (trace6.clone(), trace7.clone());
        // reset();
        let time = std::time::Instant::now();
        gkr_full_hightolow::<M, _, 5>(&mut tx, &mut trace_readset, &mut trace_writeset, &mut trace_lookupnum, &mut trace_lookupden, &mut trace_rangechecknum, &mut trace_rangecheckden, &mut trace_timechecknum, &mut trace_timecheckden);
        t_gkr_full.push(time.elapsed());
        // pi_gkr_full.push(unsafe{TRANSCRIPT.len()});

        // let mut trace_readset = trace1.clone();
        // let mut trace_writeset = trace1.iter().rev().copied().collect::<Vec<E>>();
        // let (mut trace_lookupnum, mut trace_lookupden) = (trace2.clone(), trace3.clone());
        // let (mut trace_rangechecknum, mut trace_rangecheckden) = (trace4.clone(), trace5.clone());
        // let (mut trace_timechecknum, mut trace_timecheckden) = (trace6.clone(), trace7.clone());
        // assert!(
        //     c1 == randeval_hightolow(&mut trace_readset, &z) &&
        //     c2 == randeval_hightolow(&mut trace_writeset, &z) &&
        //     c3 == randeval_hightolow(&mut trace_lookupnum, &z) &&
        //     c4 == randeval_hightolow(&mut trace_lookupden, &z) &&
        //     c5 == randeval_hightolow(&mut trace_rangechecknum, &z) &&
        //     c6 == randeval_hightolow(&mut trace_rangecheckden, &z) &&
        //     c7 == randeval_hightolow(&mut trace_timechecknum, &z) &&
        //     c8 == randeval_hightolow(&mut trace_timecheckden, &z)
        // );

        dbg!(&t_gkr_full);
        // dbg!(pi_gkr_full);
        // let diff1 = t_gkr_kernelfull[0].abs_diff(t_gkr_full[0]);
        // let diff2 = t_gkr_kernelfull[1].abs_diff(t_gkr_full[1]);
        let diffs = t_gkr_kernelfull.iter().zip(t_gkr_full.iter()).map(|(&t1,&t2)| t1.abs_diff(t2));
        let maxs = t_gkr_kernelfull.iter().zip(t_gkr_full.iter()).map(|(&t1,&t2)| t1.max(t2));
        let changes_percent_gkrfull_kernelornot = diffs.zip(maxs).map(|(diff, max)| diff.div_duration_f32(max)*100.0).collect::<Vec<f32>>();
        dbg!(changes_percent_gkrfull_kernelornot);
    }
}

// TODO: benchmark sumcheck using fully owned arrays, &, &mut..
// TODO LATER: lazy fold, buffers/antiscatter, multithreading
pub fn gkr_full_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace_readset: &mut[E], trace_writeset: &mut[E], trace_lookupnum: &mut[E], trace_lookupden: &mut[E], trace_rangechecknum: &mut[E], trace_rangecheckden: &mut[E], trace_timechecknum: &mut[E], trace_timecheckden: &mut[E]) -> (E, E, E, E, E, E, E, E, [E; M])
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace_readset.len() == 1<<M && trace_writeset.len() == 1<<M && trace_lookupnum.len() == 1<<M && trace_lookupden.len() == 1<<M);
    assert!(trace_rangechecknum.len() == 1<<M && trace_rangecheckden.len() == 1<<M && trace_timechecknum.len() == 1<<M && trace_timecheckden.len() == 1<<M);
    let mut multiacc_readset_raw = gen_multiaccprodtable_hightolow::<M, _, SKIPLAYERS>(&trace_readset);
    let mut multiacc_writeset_raw = gen_multiaccprodtable_hightolow::<M, _, SKIPLAYERS>(&trace_writeset);
    let (mut multiacc_lookupnum_raw, mut multiacc_lookupden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_lookupnum, &trace_lookupden);
    let (mut multiacc_rangechecknum_raw, mut multiacc_rangecheckden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_rangechecknum, &trace_rangecheckden);
    let (mut multiacc_timechecknum_raw, mut multiacc_timecheckden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_timechecknum, &trace_timecheckden);
    let multiacc_readset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_readset_raw, trace_readset);
    let multiacc_writeset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_writeset_raw, trace_writeset);
    let multiacc_lookupnum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupnum_raw, trace_lookupnum);
    let multiacc_lookupden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupden_raw, trace_lookupden);
    let multiacc_rangechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangechecknum_raw, trace_rangechecknum);
    let multiacc_rangecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangecheckden_raw, trace_rangecheckden);
    let multiacc_timechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timechecknum_raw, trace_timechecknum);
    let multiacc_timecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timecheckden_raw, trace_timecheckden);
    tx.send(&multiacc_readset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_writeset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupnum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangecheckden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timecheckden[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out_readset = multiacc_readset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_writeset = multiacc_writeset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_lookup = multiacc_lookupnum[SKIPLAYERS].iter().zip(multiacc_lookupden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_rangecheck = multiacc_rangechecknum[SKIPLAYERS].iter().zip(multiacc_rangecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_timecheck = multiacc_timechecknum[SKIPLAYERS].iter().zip(multiacc_timecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let info = format!("GKR OUTPUT: ({out_readset}, {out_writeset}) {out_lookup:?} {out_rangecheck:?} {out_timecheck:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i < SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim_readset, mut claim_writeset, mut claim_lookupnum, mut claim_lookupden, mut claim_rangechecknum, mut claim_rangecheckden, mut claim_timechecknum, mut claim_timecheckden) = {
        let z: &[E; SKIPLAYERS] = z.split_array_ref::<SKIPLAYERS>().0;
        // ( randeval_lowtohigh(multiacc_readset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_writeset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupnum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangecheckden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timecheckden[SKIPLAYERS], z) )
        ( randeval_hightolow(multiacc_readset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_writeset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupnum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangecheckden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timecheckden[SKIPLAYERS], z) )
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc_readset: &mut[E] = multiacc_readset[i+1].as_mut();
        let acc_writeset: &mut[E] = multiacc_writeset[i+1].as_mut();
        let acc_lookupnum: &mut[E] = multiacc_lookupnum[i+1].as_mut();
        let acc_lookupden: &mut[E] = multiacc_lookupden[i+1].as_mut();
        let acc_rangechecknum: &mut[E] = multiacc_rangechecknum[i+1].as_mut();
        let acc_rangecheckden: &mut[E] = multiacc_rangecheckden[i+1].as_mut();
        let acc_timechecknum: &mut[E] = multiacc_timechecknum[i+1].as_mut();
        let acc_timecheckden: &mut[E] = multiacc_timecheckden[i+1].as_mut();
        let accz: &mut [E; i] = z.split_array_mut::<i>().0;
        if DEBUG {
            let len = 1<<(i+1);
            debug_assert!([&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden].into_iter().all(|arr| arr.len() == len));
            println!("> gotta prove layers of len={len}:");
            dbg!(&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden);
            dbg!(&accz);
        }
        let (acc_readset_lows, acc_readset_highs) = acc_readset.split_at_mut(1<<i);
        let (acc_writeset_lows, acc_writeset_highs) = acc_writeset.split_at_mut(1<<i);
        let (acc_lookupnum_lows, acc_lookupnum_highs) = acc_lookupnum.split_at_mut(1<<i);
        let (acc_lookupden_lows, acc_lookupden_highs) = acc_lookupden.split_at_mut(1<<i);
        let (acc_rangechecknum_lows, acc_rangechecknum_highs) = acc_rangechecknum.split_at_mut(1<<i);
        let (acc_rangecheckden_lows, acc_rangecheckden_highs) = acc_rangecheckden.split_at_mut(1<<i);
        let (acc_timechecknum_lows, acc_timechecknum_highs) = acc_timechecknum.split_at_mut(1<<i);
        let (acc_timecheckden_lows, acc_timecheckden_highs) = acc_timecheckden.split_at_mut(1<<i);
        let [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7] = core::array::from_fn(|_| tx.rand());
        let claim = {
            let claim_lookup = claim_lookupnum.add_into(&claim_lookupden.mul_into(&alpha1));
            let claim_rangecheck = claim_rangechecknum.add_into(&claim_rangecheckden.mul_into(&alpha2));
            let claim_timecheck = claim_timechecknum.add_into(&claim_timecheckden.mul_into(&alpha3));
            claim_readset.add_into(&claim_writeset.mul_into(&alpha4)).add_into(&claim_lookup.mul_into(&alpha5)).add_into(&claim_rangecheck.mul_into(&alpha6)).add_into(&claim_timecheck.mul_into(&alpha7))
        };
        // sum_eqspliteqfull_lowtohigh::<_, _, 0>(tx, acc_readset_lows, acc_readset_highs, acc_writeset_lows, acc_writeset_highs, acc_lookupnum_lows, acc_lookupnum_highs, acc_lookupden_lows, acc_lookupden_highs, acc_rangechecknum_lows, acc_rangechecknum_highs, acc_rangecheckden_lows, acc_rangecheckden_highs, acc_timechecknum_lows, acc_timechecknum_highs, acc_timecheckden_lows, acc_timecheckden_highs, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, accz, claim);
        sum_eqspliteqfull_hightolow::<_, _, 0>(tx, acc_readset_lows, acc_readset_highs, acc_writeset_lows, acc_writeset_highs, acc_lookupnum_lows, acc_lookupnum_highs, acc_lookupden_lows, acc_lookupden_highs, acc_rangechecknum_lows, acc_rangechecknum_highs, acc_rangecheckden_lows, acc_rangecheckden_highs, acc_timechecknum_lows, acc_timechecknum_highs, acc_timecheckden_lows, acc_timecheckden_highs, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, accz, claim);
        // 2to1
        let mut recv = [acc_readset_lows[0], acc_readset_highs[0], acc_writeset_lows[0], acc_writeset_highs[0], acc_lookupnum_lows[0], acc_lookupnum_highs[0], acc_lookupden_lows[0], acc_lookupden_highs[0], acc_rangechecknum_lows[0], acc_rangechecknum_highs[0], acc_rangecheckden_lows[0], acc_rangecheckden_highs[0], acc_timechecknum_lows[0], acc_timechecknum_highs[0], acc_timecheckden_lows[0], acc_timecheckden_highs[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim_readset = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_writeset = randeval_lowtohigh(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupnum = randeval_lowtohigh(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupden = randeval_lowtohigh(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangechecknum = randeval_lowtohigh(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangecheckden = randeval_lowtohigh(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timechecknum = randeval_lowtohigh(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timecheckden = randeval_lowtohigh(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_readset = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_writeset = randeval_hightolow(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupnum = randeval_hightolow(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupden = randeval_hightolow(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangechecknum = randeval_hightolow(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangecheckden = randeval_hightolow(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timechecknum = randeval_hightolow(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timecheckden = randeval_hightolow(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[i] = r; // update point
    }});
    (claim_readset, claim_writeset, claim_lookupnum, claim_lookupden, claim_rangechecknum, claim_rangecheckden, claim_timechecknum, claim_timecheckden, z)
}
// TODO: benchmark sumcheck using fully owned arrays, &, &mut..
// TODO LATER: lazy fold, buffers/antiscatter, multithreading
fn gkr_full_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace_readset: &mut[E], trace_writeset: &mut[E], trace_lookupnum: &mut[E], trace_lookupden: &mut[E], trace_rangechecknum: &mut[E], trace_rangecheckden: &mut[E], trace_timechecknum: &mut[E], trace_timecheckden: &mut[E]) -> (E, E, E, E, E, E, E, E, [E; M])
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace_readset.len() == 1<<M && trace_writeset.len() == 1<<M && trace_lookupnum.len() == 1<<M && trace_lookupden.len() == 1<<M);
    assert!(trace_rangechecknum.len() == 1<<M && trace_rangecheckden.len() == 1<<M && trace_timechecknum.len() == 1<<M && trace_timecheckden.len() == 1<<M);
    let mut multiacc_readset_raw = gen_multiaccprodtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_readset);
    let mut multiacc_writeset_raw = gen_multiaccprodtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_writeset);
    let (mut multiacc_lookupnum_raw, mut multiacc_lookupden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_lookupnum, &trace_lookupden);
    let (mut multiacc_rangechecknum_raw, mut multiacc_rangecheckden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_rangechecknum, &trace_rangecheckden);
    let (mut multiacc_timechecknum_raw, mut multiacc_timecheckden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_timechecknum, &trace_timecheckden);
    let multiacc_readset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_readset_raw, trace_readset);
    let multiacc_writeset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_writeset_raw, trace_writeset);
    let multiacc_lookupnum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupnum_raw, trace_lookupnum);
    let multiacc_lookupden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupden_raw, trace_lookupden);
    let multiacc_rangechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangechecknum_raw, trace_rangechecknum);
    let multiacc_rangecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangecheckden_raw, trace_rangecheckden);
    let multiacc_timechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timechecknum_raw, trace_timechecknum);
    let multiacc_timecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timecheckden_raw, trace_timecheckden);
    tx.send(&multiacc_readset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_writeset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupnum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangecheckden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timecheckden[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out_readset = multiacc_readset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_writeset = multiacc_writeset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_lookup = multiacc_lookupnum[SKIPLAYERS].iter().zip(multiacc_lookupden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_rangecheck = multiacc_rangechecknum[SKIPLAYERS].iter().zip(multiacc_rangecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_timecheck = multiacc_timechecknum[SKIPLAYERS].iter().zip(multiacc_timecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let info = format!("GKR OUTPUT: ({out_readset}, {out_writeset}) {out_lookup:?} {out_rangecheck:?} {out_timecheck:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i>= M-SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim_readset, mut claim_writeset, mut claim_lookupnum, mut claim_lookupden, mut claim_rangechecknum, mut claim_rangecheckden, mut claim_timechecknum, mut claim_timecheckden) = {
        let z: &[E; SKIPLAYERS] = z.rsplit_array_ref::<SKIPLAYERS>().1;
        // ( randeval_lowtohigh(multiacc_readset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_writeset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupnum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangecheckden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timecheckden[SKIPLAYERS], z) )
        ( randeval_hightolow(multiacc_readset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_writeset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupnum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangecheckden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timecheckden[SKIPLAYERS], z) )
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc_readset: &mut[E] = multiacc_readset[i+1].as_mut();
        let acc_writeset: &mut[E] = multiacc_writeset[i+1].as_mut();
        let acc_lookupnum: &mut[E] = multiacc_lookupnum[i+1].as_mut();
        let acc_lookupden: &mut[E] = multiacc_lookupden[i+1].as_mut();
        let acc_rangechecknum: &mut[E] = multiacc_rangechecknum[i+1].as_mut();
        let acc_rangecheckden: &mut[E] = multiacc_rangecheckden[i+1].as_mut();
        let acc_timechecknum: &mut[E] = multiacc_timechecknum[i+1].as_mut();
        let acc_timecheckden: &mut[E] = multiacc_timecheckden[i+1].as_mut();
        let accz: &mut [E; i] = z.rsplit_array_mut::<i>().1;
        if DEBUG {
            let len = 1<<(i+1);
            debug_assert!([&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden].into_iter().all(|arr| arr.len() == len));
            println!("> gotta prove layers of len={len}:");
            dbg!(&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden);
            dbg!(&accz);
        }
        let mut acc_readset_evens = acc_readset.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_readset_odds = acc_readset.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_writeset_evens = acc_writeset.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_writeset_odds = acc_writeset.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupnum_evens = acc_lookupnum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupnum_odds = acc_lookupnum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupden_evens = acc_lookupden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupden_odds = acc_lookupden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangechecknum_evens = acc_rangechecknum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangechecknum_odds = acc_rangechecknum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangecheckden_evens = acc_rangecheckden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangecheckden_odds = acc_rangecheckden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timechecknum_evens = acc_timechecknum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timechecknum_odds = acc_timechecknum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timecheckden_evens = acc_timecheckden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timecheckden_odds = acc_timecheckden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7] = core::array::from_fn(|_| tx.rand());
        let claim = {
            let claim_lookup = claim_lookupnum.add_into(&claim_lookupden.mul_into(&alpha1));
            let claim_rangecheck = claim_rangechecknum.add_into(&claim_rangecheckden.mul_into(&alpha2));
            let claim_timecheck = claim_timechecknum.add_into(&claim_timecheckden.mul_into(&alpha3));
            claim_readset.add_into(&claim_writeset.mul_into(&alpha4)).add_into(&claim_lookup.mul_into(&alpha5)).add_into(&claim_rangecheck.mul_into(&alpha6)).add_into(&claim_timecheck.mul_into(&alpha7))
        };
        sum_eqspliteqfull_lowtohigh::<_, _, 0>(tx, &mut acc_readset_evens, &mut acc_readset_odds, &mut acc_writeset_evens, &mut acc_writeset_odds, &mut acc_lookupnum_evens, &mut acc_lookupnum_odds, &mut acc_lookupden_evens, &mut acc_lookupden_odds, &mut acc_rangechecknum_evens, &mut acc_rangechecknum_odds, &mut acc_rangecheckden_evens, &mut acc_rangecheckden_odds, &mut acc_timechecknum_evens, &mut acc_timechecknum_odds, &mut acc_timecheckden_evens, &mut acc_timecheckden_odds, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, accz, claim);
        // sum_eqspliteqfull_hightolow::<_, _, 0>(tx, &mut acc_readset_evens, &mut acc_readset_odds, &mut acc_writeset_evens, &mut acc_writeset_odds, &mut acc_lookupnum_evens, &mut acc_lookupnum_odds, &mut acc_lookupden_evens, &mut acc_lookupden_odds, &mut acc_rangechecknum_evens, &mut acc_rangechecknum_odds, &mut acc_rangecheckden_evens, &mut acc_rangecheckden_odds, &mut acc_timechecknum_evens, &mut acc_timechecknum_odds, &mut acc_timecheckden_evens, &mut acc_timecheckden_odds, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, accz, claim);
        // 2to1
        let mut recv = [acc_readset_evens[0], acc_readset_odds[0], acc_writeset_evens[0], acc_writeset_odds[0], acc_lookupnum_evens[0], acc_lookupnum_odds[0], acc_lookupden_evens[0], acc_lookupden_odds[0], acc_rangechecknum_evens[0], acc_rangechecknum_odds[0], acc_rangecheckden_evens[0], acc_rangecheckden_odds[0], acc_timechecknum_evens[0], acc_timechecknum_odds[0], acc_timecheckden_evens[0], acc_timecheckden_odds[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim_readset = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_writeset = randeval_lowtohigh(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupnum = randeval_lowtohigh(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupden = randeval_lowtohigh(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangechecknum = randeval_lowtohigh(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangecheckden = randeval_lowtohigh(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timechecknum = randeval_lowtohigh(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timecheckden = randeval_lowtohigh(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_readset = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_writeset = randeval_hightolow(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupnum = randeval_hightolow(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupden = randeval_hightolow(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangechecknum = randeval_hightolow(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangecheckden = randeval_hightolow(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timechecknum = randeval_hightolow(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timecheckden = randeval_hightolow(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[M-i-1] = r; // update point
    }});
    (claim_readset, claim_writeset, claim_lookupnum, claim_lookupden, claim_rangechecknum, claim_rangecheckden, claim_timechecknum, claim_timecheckden, z)
}
// TODO: benchmark sumcheck using fully owned arrays, &, &mut..
// TODO LATER: lazy fold, buffers/antiscatter, multithreading
fn gkr_kernelfull_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace_readset: &mut[E], trace_writeset: &mut[E], trace_lookupnum: &mut[E], trace_lookupden: &mut[E], trace_rangechecknum: &mut[E], trace_rangecheckden: &mut[E], trace_timechecknum: &mut[E], trace_timecheckden: &mut[E]) -> (E, E, E, E, E, E, E, E, [E; M])
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_glinquad_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], &mut[alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7]: &mut[E; 3+4], (ie, io): (usize, usize)) -> (E, E) {
    // NB: even/low odd/high depending on index and folding order
        #[inline(always)]
        #[allow(non_snake_case)]
        fn product_glinquad_operation<E: FEXT>([trace1, trace2]: &mut [&mut[E]; 2], []: &mut[E;0], (ie, io):(usize, usize)) -> (E, E) {
            // A*B
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            trace1[io] = subA;
            trace2[io] = subB;
            let lin = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
            let quad = subA.mul_into(&subB);
            (lin, quad)
        }
        #[inline(always)]
        #[allow(non_snake_case)]
        fn logup_glinquad_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut[E]; 4], &mut[alpha]: &mut[E;1], (ie, io):(usize, usize)) -> (E, E) {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let (evenC, oddC) = (trace3[ie], trace3[io]);
            let (evenD, oddD) = (trace4[ie], trace4[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            let subC = oddC.sub_into(&evenC);
            let subD = oddD.sub_into(&evenD);
            trace1[io] = subA;
            trace2[io] = subB;
            trace3[io] = subC;
            trace4[io] = subD;
            let evenRLC = evenA.add_into(&evenB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let subRLC = subA.add_into(&subB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let lin = evenD.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subD)).add_into(&evenB.mul_into(&subC)).add_into(&evenC.mul_into(&subB));
            let quad = subD.mul_into(&subRLC).add_into(&subB.mul_into(&subC));
            (lin, quad)
        }
        let (lin_readset, quad_readset) = product_glinquad_operation(&mut[trace_readset0, trace_readset1], &mut[], (ie, io));
        let (lin_writeset, quad_writeset) = product_glinquad_operation(&mut[trace_writeset0, trace_writeset1], &mut[], (ie, io));
        let (lin_lookup, quad_lookup) = logup_glinquad_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], &mut[alpha1], (ie, io));
        let (lin_rangecheck, quad_rangecheck) = logup_glinquad_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], &mut[alpha2], (ie, io));
        let (lin_timecheck, quad_timecheck) = logup_glinquad_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], &mut[alpha3], (ie, io));
        let lin = lin_readset.add_into(&lin_writeset.mul_into(&alpha4)).add_into(&lin_lookup.mul_into(&alpha5)).add_into(&lin_rangecheck.mul_into(&alpha6)).add_into(&lin_timecheck.mul_into(&alpha7));
        let quad = quad_readset.add_into(&quad_writeset.mul_into(&alpha4)).add_into(&quad_lookup.mul_into(&alpha5)).add_into(&quad_rangecheck.mul_into(&alpha6)).add_into(&quad_timecheck.mul_into(&alpha7));
        (lin, quad)
    }
    #[inline(always)]
    fn full_point_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], &mut[alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7]: &mut[E; 3+4], i: usize) -> E {
        fn product_point_operation<E: FEXT>([trace1, trace2]: &mut [&mut [E]; 2], &mut[]: &mut[E; 0], i: usize) -> E {
            // A*B
            trace1[i].mul_into(&trace2[i])
        }
        fn logup_point_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], i: usize) -> E {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let tmp = trace3[i].mul_into(&trace2[i]);
            trace4[i].mul_into(&trace1[i].add_into(&trace2[i].mul_into(&alpha))).add_into(&tmp)
        }
        let point_readset = product_point_operation(&mut[trace_readset0, trace_readset1], &mut[], i);
        let point_writeset = product_point_operation(&mut[trace_writeset0, trace_writeset1], &mut[], i);
        let point_lookup = logup_point_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], &mut[alpha1], i);
        let point_rangecheck = logup_point_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], &mut[alpha2], i);
        let point_timecheck = logup_point_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], &mut[alpha3], i);
        let point = point_readset.add_into(&point_writeset.mul_into(&alpha4)).add_into(&point_lookup.mul_into(&alpha5)).add_into(&point_rangecheck.mul_into(&alpha6)).add_into(&point_timecheck.mul_into(&alpha7));
        point
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_fold_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], r: E, (ie, io): (usize, usize)) {
    // NB: even/low odd/high depending on index and folding order
        fn product_fold_operation<E: FEXT>([trace1, trace2]: &mut [&mut [E]; 2], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        fn logup_fold_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        product_fold_operation(&mut[trace_readset0, trace_readset1], r, (ie, io));
        product_fold_operation(&mut[trace_writeset0, trace_writeset1], r, (ie, io));
        logup_fold_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], r, (ie, io));
        logup_fold_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], r, (ie, io));
        logup_fold_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], r, (ie, io));
    }
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace_readset.len() == 1<<M && trace_writeset.len() == 1<<M && trace_lookupnum.len() == 1<<M && trace_lookupden.len() == 1<<M);
    assert!(trace_rangechecknum.len() == 1<<M && trace_rangecheckden.len() == 1<<M && trace_timechecknum.len() == 1<<M && trace_timecheckden.len() == 1<<M);
    let mut multiacc_readset_raw = gen_multiaccprodtable_hightolow::<M, _, SKIPLAYERS>(&trace_readset);
    let mut multiacc_writeset_raw = gen_multiaccprodtable_hightolow::<M, _, SKIPLAYERS>(&trace_writeset);
    let (mut multiacc_lookupnum_raw, mut multiacc_lookupden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_lookupnum, &trace_lookupden);
    let (mut multiacc_rangechecknum_raw, mut multiacc_rangecheckden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_rangechecknum, &trace_rangecheckden);
    let (mut multiacc_timechecknum_raw, mut multiacc_timecheckden_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace_timechecknum, &trace_timecheckden);
    let multiacc_readset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_readset_raw, trace_readset);
    let multiacc_writeset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_writeset_raw, trace_writeset);
    let multiacc_lookupnum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupnum_raw, trace_lookupnum);
    let multiacc_lookupden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupden_raw, trace_lookupden);
    let multiacc_rangechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangechecknum_raw, trace_rangechecknum);
    let multiacc_rangecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangecheckden_raw, trace_rangecheckden);
    let multiacc_timechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timechecknum_raw, trace_timechecknum);
    let multiacc_timecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timecheckden_raw, trace_timecheckden);
    tx.send(&multiacc_readset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_writeset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupnum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangecheckden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timecheckden[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out_readset = multiacc_readset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_writeset = multiacc_writeset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_lookup = multiacc_lookupnum[SKIPLAYERS].iter().zip(multiacc_lookupden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_rangecheck = multiacc_rangechecknum[SKIPLAYERS].iter().zip(multiacc_rangecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_timecheck = multiacc_timechecknum[SKIPLAYERS].iter().zip(multiacc_timecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let info = format!("GKR OUTPUT: ({out_readset}, {out_writeset}) {out_lookup:?} {out_rangecheck:?} {out_timecheck:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i < SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim_readset, mut claim_writeset, mut claim_lookupnum, mut claim_lookupden, mut claim_rangechecknum, mut claim_rangecheckden, mut claim_timechecknum, mut claim_timecheckden) = {
        let z: &[E; SKIPLAYERS] = z.split_array_ref::<SKIPLAYERS>().0;
        // ( randeval_lowtohigh(multiacc_readset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_writeset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupnum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangecheckden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timecheckden[SKIPLAYERS], z) )
        ( randeval_hightolow(multiacc_readset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_writeset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupnum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangecheckden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timecheckden[SKIPLAYERS], z) )
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc_readset: &mut[E] = multiacc_readset[i+1].as_mut();
        let acc_writeset: &mut[E] = multiacc_writeset[i+1].as_mut();
        let acc_lookupnum: &mut[E] = multiacc_lookupnum[i+1].as_mut();
        let acc_lookupden: &mut[E] = multiacc_lookupden[i+1].as_mut();
        let acc_rangechecknum: &mut[E] = multiacc_rangechecknum[i+1].as_mut();
        let acc_rangecheckden: &mut[E] = multiacc_rangecheckden[i+1].as_mut();
        let acc_timechecknum: &mut[E] = multiacc_timechecknum[i+1].as_mut();
        let acc_timecheckden: &mut[E] = multiacc_timecheckden[i+1].as_mut();
        let accz: &mut [E; i] = z.split_array_mut::<i>().0;
        if DEBUG {
            let len = 1<<(i+1);
            debug_assert!([&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden].into_iter().all(|arr| arr.len() == len));
            println!("> gotta prove layers of len={len}:");
            dbg!(&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden);
            dbg!(&accz);
        }
        let (acc_readset_lows, acc_readset_highs) = acc_readset.split_at_mut(1<<i);
        let (acc_writeset_lows, acc_writeset_highs) = acc_writeset.split_at_mut(1<<i);
        let (acc_lookupnum_lows, acc_lookupnum_highs) = acc_lookupnum.split_at_mut(1<<i);
        let (acc_lookupden_lows, acc_lookupden_highs) = acc_lookupden.split_at_mut(1<<i);
        let (acc_rangechecknum_lows, acc_rangechecknum_highs) = acc_rangechecknum.split_at_mut(1<<i);
        let (acc_rangecheckden_lows, acc_rangecheckden_highs) = acc_rangecheckden.split_at_mut(1<<i);
        let (acc_timechecknum_lows, acc_timechecknum_highs) = acc_timechecknum.split_at_mut(1<<i);
        let (acc_timecheckden_lows, acc_timecheckden_highs) = acc_timecheckden.split_at_mut(1<<i);
        let [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7] = core::array::from_fn(|_| tx.rand());
        let claim = {
            let claim_lookup = claim_lookupnum.add_into(&claim_lookupden.mul_into(&alpha1));
            let claim_rangecheck = claim_rangechecknum.add_into(&claim_rangecheckden.mul_into(&alpha2));
            let claim_timecheck = claim_timechecknum.add_into(&claim_timecheckden.mul_into(&alpha3));
            claim_readset.add_into(&claim_writeset.mul_into(&alpha4)).add_into(&claim_lookup.mul_into(&alpha5)).add_into(&claim_rangecheck.mul_into(&alpha6)).add_into(&claim_timecheck.mul_into(&alpha7))
        };
        // let traces = [acc_readset_lows.as_mut_slice(), acc_readset_highs.as_mut_slice(), acc_writeset_lows.as_mut_slice(), acc_writeset_highs.as_mut_slice(), acc_lookupnum_lows.as_mut_slice(), acc_lookupnum_highs.as_mut_slice(), acc_lookupden_lows.as_mut_slice(), acc_lookupden_highs.as_mut_slice(), acc_rangechecknum_lows.as_mut_slice(), acc_rangechecknum_highs.as_mut_slice(), acc_rangecheckden_lows.as_mut_slice(), acc_rangecheckden_highs.as_mut_slice(), acc_timechecknum_lows.as_mut_slice(), acc_timechecknum_highs.as_mut_slice(), acc_timecheckden_lows.as_mut_slice(), acc_timecheckden_highs.as_mut_slice()];
        let traces = [&mut acc_readset_lows[..], &mut acc_readset_highs[..], &mut acc_writeset_lows[..], &mut acc_writeset_highs[..], &mut acc_lookupnum_lows[..], &mut acc_lookupnum_highs[..], &mut acc_lookupden_lows[..], &mut acc_lookupden_highs[..], &mut acc_rangechecknum_lows[..], &mut acc_rangechecknum_highs[..], &mut acc_rangecheckden_lows[..], &mut acc_rangecheckden_highs[..], &mut acc_timechecknum_lows[..], &mut acc_timechecknum_highs[..], &mut acc_timecheckden_lows[..], &mut acc_timecheckden_highs[..]];
        // sum_eqspliteqKERNEL_lowtohigh::<_, _, _, _, _, _, _, 0>(tx, full_glinquad_operation, full_point_operation, full_fold_operation, traces, [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7], accz, claim);
        sum_eqspliteqKERNEL_hightolow::<_, _, _, _, _, _, _, 0>(tx, full_glinquad_operation, full_point_operation, full_fold_operation, traces, [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7], accz, claim);
        // 2to1
        let mut recv = [acc_readset_lows[0], acc_readset_highs[0], acc_writeset_lows[0], acc_writeset_highs[0], acc_lookupnum_lows[0], acc_lookupnum_highs[0], acc_lookupden_lows[0], acc_lookupden_highs[0], acc_rangechecknum_lows[0], acc_rangechecknum_highs[0], acc_rangecheckden_lows[0], acc_rangecheckden_highs[0], acc_timechecknum_lows[0], acc_timechecknum_highs[0], acc_timecheckden_lows[0], acc_timecheckden_highs[0]];
        tx.send(&recv);
        let r: E = tx.rand();
        // claim_readset = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_writeset = randeval_lowtohigh(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupnum = randeval_lowtohigh(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupden = randeval_lowtohigh(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangechecknum = randeval_lowtohigh(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangecheckden = randeval_lowtohigh(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timechecknum = randeval_lowtohigh(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timecheckden = randeval_lowtohigh(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_readset = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_writeset = randeval_hightolow(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupnum = randeval_hightolow(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupden = randeval_hightolow(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangechecknum = randeval_hightolow(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangecheckden = randeval_hightolow(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timechecknum = randeval_hightolow(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timecheckden = randeval_hightolow(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[i] = r; // update point
    }});
    (claim_readset, claim_writeset, claim_lookupnum, claim_lookupden, claim_rangechecknum, claim_rangecheckden, claim_timechecknum, claim_timecheckden, z)
}
// TODO: benchmark sumcheck using fully owned arrays, &, &mut..
// TODO LATER: lazy fold, buffers/antiscatter, multithreading
fn gkr_kernelfull_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace_readset: &mut[E], trace_writeset: &mut[E], trace_lookupnum: &mut[E], trace_lookupden: &mut[E], trace_rangechecknum: &mut[E], trace_rangecheckden: &mut[E], trace_timechecknum: &mut[E], trace_timecheckden: &mut[E]) -> (E, E, E, E, E, E, E, E, [E; M])
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_glinquad_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], &mut[alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7]: &mut[E; 3+4], (ie, io): (usize, usize)) -> (E, E) {
    // NB: even/low odd/high depending on index and folding order
        #[inline(always)]
        #[allow(non_snake_case)]
        fn product_glinquad_operation<E: FEXT>([trace1, trace2]: &mut [&mut[E]; 2], []: &mut[E;0], (ie, io):(usize, usize)) -> (E, E) {
            // A*B
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            trace1[io] = subA;
            trace2[io] = subB;
            let lin = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
            let quad = subA.mul_into(&subB);
            (lin, quad)
        }
        #[inline(always)]
        #[allow(non_snake_case)]
        fn logup_glinquad_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut[E]; 4], &mut[alpha]: &mut[E;1], (ie, io):(usize, usize)) -> (E, E) {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let (evenC, oddC) = (trace3[ie], trace3[io]);
            let (evenD, oddD) = (trace4[ie], trace4[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            let subC = oddC.sub_into(&evenC);
            let subD = oddD.sub_into(&evenD);
            trace1[io] = subA;
            trace2[io] = subB;
            trace3[io] = subC;
            trace4[io] = subD;
            let evenRLC = evenA.add_into(&evenB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let subRLC = subA.add_into(&subB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let lin = evenD.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subD)).add_into(&evenB.mul_into(&subC)).add_into(&evenC.mul_into(&subB));
            let quad = subD.mul_into(&subRLC).add_into(&subB.mul_into(&subC));
            (lin, quad)
        }
        let (lin_readset, quad_readset) = product_glinquad_operation(&mut[trace_readset0, trace_readset1], &mut[], (ie, io));
        let (lin_writeset, quad_writeset) = product_glinquad_operation(&mut[trace_writeset0, trace_writeset1], &mut[], (ie, io));
        let (lin_lookup, quad_lookup) = logup_glinquad_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], &mut[alpha1], (ie, io));
        let (lin_rangecheck, quad_rangecheck) = logup_glinquad_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], &mut[alpha2], (ie, io));
        let (lin_timecheck, quad_timecheck) = logup_glinquad_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], &mut[alpha3], (ie, io));
        let lin = lin_readset.add_into(&lin_writeset.mul_into(&alpha4)).add_into(&lin_lookup.mul_into(&alpha5)).add_into(&lin_rangecheck.mul_into(&alpha6)).add_into(&lin_timecheck.mul_into(&alpha7));
        let quad = quad_readset.add_into(&quad_writeset.mul_into(&alpha4)).add_into(&quad_lookup.mul_into(&alpha5)).add_into(&quad_rangecheck.mul_into(&alpha6)).add_into(&quad_timecheck.mul_into(&alpha7));
        (lin, quad)
    }
    #[inline(always)]
    fn full_point_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], &mut[alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7]: &mut[E; 3+4], i: usize) -> E {
        fn product_point_operation<E: FEXT>([trace1, trace2]: &mut [&mut [E]; 2], &mut[]: &mut[E; 0], i: usize) -> E {
            // A*B
            trace1[i].mul_into(&trace2[i])
        }
        fn logup_point_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], i: usize) -> E {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let tmp = trace3[i].mul_into(&trace2[i]);
            trace4[i].mul_into(&trace1[i].add_into(&trace2[i].mul_into(&alpha))).add_into(&tmp)
        }
        let point_readset = product_point_operation(&mut[trace_readset0, trace_readset1], &mut[], i);
        let point_writeset = product_point_operation(&mut[trace_writeset0, trace_writeset1], &mut[], i);
        let point_lookup = logup_point_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], &mut[alpha1], i);
        let point_rangecheck = logup_point_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], &mut[alpha2], i);
        let point_timecheck = logup_point_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], &mut[alpha3], i);
        let point = point_readset.add_into(&point_writeset.mul_into(&alpha4)).add_into(&point_lookup.mul_into(&alpha5)).add_into(&point_rangecheck.mul_into(&alpha6)).add_into(&point_timecheck.mul_into(&alpha7));
        point
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_fold_operation<E: FEXT>([trace_readset0, trace_readset1, trace_writeset0, trace_writeset1, trace_lookupnum0, trace_lookupnum1, trace_lookupden0, trace_lookupden1, trace_rangechecknum0, trace_rangechecknum1, trace_rangecheckden0, trace_rangecheckden1, trace_timechecknum0, trace_timechecknum1, trace_timecheckden0, trace_timecheckden1]: &mut [&mut [E]; 8*2], r: E, (ie, io): (usize, usize)) {
    // NB: even/low odd/high depending on index and folding order
        fn product_fold_operation<E: FEXT>([trace1, trace2]: &mut [&mut [E]; 2], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        fn logup_fold_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        product_fold_operation(&mut[trace_readset0, trace_readset1], r, (ie, io));
        product_fold_operation(&mut[trace_writeset0, trace_writeset1], r, (ie, io));
        logup_fold_operation(&mut[trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1], r, (ie, io));
        logup_fold_operation(&mut[trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1], r, (ie, io));
        logup_fold_operation(&mut[trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1], r, (ie, io));
    }
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace_readset.len() == 1<<M && trace_writeset.len() == 1<<M && trace_lookupnum.len() == 1<<M && trace_lookupden.len() == 1<<M);
    assert!(trace_rangechecknum.len() == 1<<M && trace_rangecheckden.len() == 1<<M && trace_timechecknum.len() == 1<<M && trace_timecheckden.len() == 1<<M);
    let mut multiacc_readset_raw = gen_multiaccprodtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_readset);
    let mut multiacc_writeset_raw = gen_multiaccprodtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_writeset);
    let (mut multiacc_lookupnum_raw, mut multiacc_lookupden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_lookupnum, &trace_lookupden);
    let (mut multiacc_rangechecknum_raw, mut multiacc_rangecheckden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_rangechecknum, &trace_rangecheckden);
    let (mut multiacc_timechecknum_raw, mut multiacc_timecheckden_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace_timechecknum, &trace_timecheckden);
    let multiacc_readset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_readset_raw, trace_readset);
    let multiacc_writeset: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_writeset_raw, trace_writeset);
    let multiacc_lookupnum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupnum_raw, trace_lookupnum);
    let multiacc_lookupden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_lookupden_raw, trace_lookupden);
    let multiacc_rangechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangechecknum_raw, trace_rangechecknum);
    let multiacc_rangecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_rangecheckden_raw, trace_rangecheckden);
    let multiacc_timechecknum: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timechecknum_raw, trace_timechecknum);
    let multiacc_timecheckden: [&mut[E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_timecheckden_raw, trace_timecheckden);
    tx.send(&multiacc_readset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_writeset[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupnum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_lookupden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_rangecheckden[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timechecknum[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc_timecheckden[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out_readset = multiacc_readset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_writeset = multiacc_writeset[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let out_lookup = multiacc_lookupnum[SKIPLAYERS].iter().zip(multiacc_lookupden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_rangecheck = multiacc_rangechecknum[SKIPLAYERS].iter().zip(multiacc_rangecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let out_timecheck = multiacc_timechecknum[SKIPLAYERS].iter().zip(multiacc_timecheckden[SKIPLAYERS].iter()).fold((E::ZERO, E::ONE), |(accnum, accden), (&num, &den)|
             (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        );
        let info = format!("GKR OUTPUT: ({out_readset}, {out_writeset}) {out_lookup:?} {out_rangecheck:?} {out_timecheck:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i>= M-SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim_readset, mut claim_writeset, mut claim_lookupnum, mut claim_lookupden, mut claim_rangechecknum, mut claim_rangecheckden, mut claim_timechecknum, mut claim_timecheckden) = {
        let z: &[E; SKIPLAYERS] = z.rsplit_array_ref::<SKIPLAYERS>().1;
        // ( randeval_lowtohigh(multiacc_readset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_writeset[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupnum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_lookupden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_rangecheckden[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timechecknum[SKIPLAYERS], z),
        //     randeval_lowtohigh(multiacc_timecheckden[SKIPLAYERS], z) )
        ( randeval_hightolow(multiacc_readset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_writeset[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupnum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_lookupden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_rangecheckden[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timechecknum[SKIPLAYERS], z),
            randeval_hightolow(multiacc_timecheckden[SKIPLAYERS], z) )
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc_readset: &mut[E] = multiacc_readset[i+1].as_mut();
        let acc_writeset: &mut[E] = multiacc_writeset[i+1].as_mut();
        let acc_lookupnum: &mut[E] = multiacc_lookupnum[i+1].as_mut();
        let acc_lookupden: &mut[E] = multiacc_lookupden[i+1].as_mut();
        let acc_rangechecknum: &mut[E] = multiacc_rangechecknum[i+1].as_mut();
        let acc_rangecheckden: &mut[E] = multiacc_rangecheckden[i+1].as_mut();
        let acc_timechecknum: &mut[E] = multiacc_timechecknum[i+1].as_mut();
        let acc_timecheckden: &mut[E] = multiacc_timecheckden[i+1].as_mut();
        let accz: &mut [E; i] = z.rsplit_array_mut::<i>().1;
        if DEBUG {
            let len = 1<<(i+1);
            debug_assert!([&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden].into_iter().all(|arr| arr.len() == len));
            println!("> gotta prove layers of len={len}:");
            dbg!(&acc_readset, &acc_writeset, &acc_lookupnum, &acc_lookupden, &acc_rangechecknum, &acc_rangecheckden, &acc_timechecknum, &acc_timecheckden);
            dbg!(&accz);
        }
        let mut acc_readset_evens = acc_readset.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_readset_odds = acc_readset.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_writeset_evens = acc_writeset.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_writeset_odds = acc_writeset.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupnum_evens = acc_lookupnum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupnum_odds = acc_lookupnum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupden_evens = acc_lookupden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_lookupden_odds = acc_lookupden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangechecknum_evens = acc_rangechecknum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangechecknum_odds = acc_rangechecknum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangecheckden_evens = acc_rangecheckden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_rangecheckden_odds = acc_rangecheckden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timechecknum_evens = acc_timechecknum.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timechecknum_odds = acc_timechecknum.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timecheckden_evens = acc_timecheckden.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_timecheckden_odds = acc_timecheckden.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7] = core::array::from_fn(|_| tx.rand());
        let claim = {
            let claim_lookup = claim_lookupnum.add_into(&claim_lookupden.mul_into(&alpha1));
            let claim_rangecheck = claim_rangechecknum.add_into(&claim_rangecheckden.mul_into(&alpha2));
            let claim_timecheck = claim_timechecknum.add_into(&claim_timecheckden.mul_into(&alpha3));
            claim_readset.add_into(&claim_writeset.mul_into(&alpha4)).add_into(&claim_lookup.mul_into(&alpha5)).add_into(&claim_rangecheck.mul_into(&alpha6)).add_into(&claim_timecheck.mul_into(&alpha7))
        };
        let traces = [acc_readset_evens.as_mut_slice(), acc_readset_odds.as_mut_slice(), acc_writeset_evens.as_mut_slice(), acc_writeset_odds.as_mut_slice(), acc_lookupnum_evens.as_mut_slice(), acc_lookupnum_odds.as_mut_slice(), acc_lookupden_evens.as_mut_slice(), acc_lookupden_odds.as_mut_slice(), acc_rangechecknum_evens.as_mut_slice(), acc_rangechecknum_odds.as_mut_slice(), acc_rangecheckden_evens.as_mut_slice(), acc_rangecheckden_odds.as_mut_slice(), acc_timechecknum_evens.as_mut_slice(), acc_timechecknum_odds.as_mut_slice(), acc_timecheckden_evens.as_mut_slice(), acc_timecheckden_odds.as_mut_slice()];
        sum_eqspliteqKERNEL_lowtohigh::<_, _, _, _, _, _, _, 0>(tx, full_glinquad_operation, full_point_operation, full_fold_operation, traces, [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7], accz, claim);
        // sum_eqspliteqKERNEL_hightolow::<_, _, _, _, _, _, _, 0>(tx, full_glinquad_operation, full_point_operation, full_fold_operation, traces, [alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7], accz, claim);
        // 2to1
        let mut recv = [acc_readset_evens[0], acc_readset_odds[0], acc_writeset_evens[0], acc_writeset_odds[0], acc_lookupnum_evens[0], acc_lookupnum_odds[0], acc_lookupden_evens[0], acc_lookupden_odds[0], acc_rangechecknum_evens[0], acc_rangechecknum_odds[0], acc_rangecheckden_evens[0], acc_rangecheckden_odds[0], acc_timechecknum_evens[0], acc_timechecknum_odds[0], acc_timecheckden_evens[0], acc_timecheckden_odds[0]];
        tx.send(&recv);
        let r: E = tx.rand();
        // claim_readset = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_writeset = randeval_lowtohigh(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupnum = randeval_lowtohigh(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_lookupden = randeval_lowtohigh(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangechecknum = randeval_lowtohigh(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_rangecheckden = randeval_lowtohigh(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timechecknum = randeval_lowtohigh(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim_timecheckden = randeval_lowtohigh(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_readset = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_writeset = randeval_hightolow(&mut recv[2..4], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupnum = randeval_hightolow(&mut recv[4..6], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_lookupden = randeval_hightolow(&mut recv[6..8], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangechecknum = randeval_hightolow(&mut recv[8..10], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_rangecheckden = randeval_hightolow(&mut recv[10..12], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timechecknum = randeval_hightolow(&mut recv[12..14], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim_timecheckden = randeval_hightolow(&mut recv[14..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[M-i-1] = r; // update point
    }});
    (claim_readset, claim_writeset, claim_lookupnum, claim_lookupden, claim_rangechecknum, claim_rangecheckden, claim_timechecknum, claim_timecheckden, z)
}
fn gkr_kernelgrandsum_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn logup_g_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], (ie, io): (usize, usize)) -> (E, E) {
        // NB: even/low odd/high depending on index and folding order
        let (evenA, oddA) = (trace1[ie], trace1[io]);
        let (evenB, oddB) = (trace2[ie], trace2[io]);
        let (evenC, oddC) = (trace3[ie], trace3[io]);
        let (evenD, oddD) = (trace4[ie], trace4[io]);
        let subA = oddA.sub_into(&evenA);
        let subB = oddB.sub_into(&evenB);
        let subC = oddC.sub_into(&evenC);
        let subD = oddD.sub_into(&evenD);
        trace1[io] = subA;
        trace2[io] = subB;
        trace3[io] = subC;
        trace4[io] =subD;
        let evenRLC = evenA.add_into(&evenD.clone().mul_into(&alpha));
        let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
        let lin = evenB.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subB)).add_into(&evenC.mul_into(&subD)).add_into(&evenD.mul_into(&subC));
        let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
        (lin, quad)
    }
    #[inline(always)]
    fn logup_p_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], i: usize) -> E {
        let operation = {
            let tmp = trace3[i].mul_into(&trace4[i]);
            trace2[i].mul_into(&trace1[i].add_into(&trace4[i].mul_into(&alpha))).add_into(&tmp)
        };
        // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
        operation
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn logup_fold_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], r: E, (ie, io): (usize, usize)) {
        // NB: even/low odd/high depending on index and folding order
        let (lowA, subA) = (trace1[ie], trace1[io]);
        let (lowB, subB) = (trace2[ie], trace2[io]);
        let (lowC, subC) = (trace3[ie], trace3[io]);
        let (lowD, subD) = (trace4[ie], trace4[io]);
        let foldA = lowA.add_into(&subA.mul_into(&r));
        let foldB = lowB.add_into(&subB.mul_into(&r));
        let foldC = lowC.add_into(&subC.mul_into(&r));
        let foldD = lowD.add_into(&subD.mul_into(&r));
        trace1[ie] = foldA;
        trace2[ie] = foldB;
        trace3[ie] = foldC;
        trace4[ie] = foldD;
    }
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let (mut multiacc1_raw, mut multiacc2_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace1, &trace2);
    let multiacc1: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc1_raw, trace1);
    let multiacc2: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc2_raw, trace2);
    tx.send(&multiacc1[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc2[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out = multiacc1[SKIPLAYERS].iter().zip(multiacc2[SKIPLAYERS].iter()).fold((E::ZERO,E::ONE), |(accnum, accden), (&num, &den)| {
            (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        });
        let info = format!("GKR OUTPUT: {out:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i < SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim1, mut claim2) = {
        let z: &[E; SKIPLAYERS] = z.split_array_ref::<SKIPLAYERS>().0;
        // (randeval_lowtohigh(multiacc1[SKIPLAYERS], z), randeval_lowtohigh(multiacc2[SKIPLAYERS], z))
        (randeval_hightolow(multiacc1[SKIPLAYERS], z), randeval_hightolow(multiacc2[SKIPLAYERS], z))
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc1: &mut[E] = multiacc1[i+1].as_mut();
        let acc2: &mut[E] = multiacc2[i+1].as_mut();
        let accz: &mut [E; i] = z.split_array_mut::<i>().0;
        if DEBUG {
            let len = 1<<(i+1);
            debug_assert!(len == acc1.len() && len == acc2.len());
            println!("> gotta prove layer of len={len}: {acc1:?} and {acc2:?}");
            dbg!(&accz);
        }
        let (acc1_lows, acc1_highs) = acc1.split_at_mut(1<<i);
        let (acc2_lows, acc2_highs) = acc2.split_at_mut(1<<i);
        let alpha = tx.rand();
        let claim = claim1.add_into(&claim2.mul_into(&alpha));
        let traces = [&mut acc1_lows[..], &mut acc2_highs[..], &mut acc1_highs[..], &mut acc2_lows[..]];
        // sum_eqspliteqKERNEL_lowtohigh::<_, _, _, _, _, 0>(tx, logup_g_operation, logup_p_operation, traces, [alpha], accz, claim);
        sum_eqspliteqKERNEL_hightolow::<_, _, _, _, _, _, _, 0>(tx, logup_g_operation, logup_p_operation, logup_fold_operation, traces, [alpha], accz, claim);
        // 2to1
        let mut recv = [acc1_lows[0], acc1_highs[0], acc2_lows[0], acc2_highs[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim1 = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim2 = randeval_lowtohigh(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim1 = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim2 = randeval_hightolow(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[i] = r; // update point
    }});
}
// TODO: benchmark sumcheck using fully owned arrays, &, &mut..
// TODO: add associated constants so u can have variable amount of ram/logup
// TODO LATER: lazy fold, buffers/antiscatter, multithreading
fn gkr_kernelgrandsum_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn logup_g_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], (ie, io): (usize, usize)) -> (E, E) {
        // NB: even/low odd/high depending on index and folding order
        let (evenA, oddA) = (trace1[ie], trace1[io]);
        let (evenB, oddB) = (trace2[ie], trace2[io]);
        let (evenC, oddC) = (trace3[ie], trace3[io]);
        let (evenD, oddD) = (trace4[ie], trace4[io]);
        let subA = oddA.sub_into(&evenA);
        let subB = oddB.sub_into(&evenB);
        let subC = oddC.sub_into(&evenC);
        let subD = oddD.sub_into(&evenD);
        trace1[io] = subA;
        trace2[io] = subB;
        trace3[io] = subC;
        trace4[io] =subD;
        let evenRLC = evenA.add_into(&evenD.clone().mul_into(&alpha));
        let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
        let lin = evenB.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subB)).add_into(&evenC.mul_into(&subD)).add_into(&evenD.mul_into(&subC));
        let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
        (lin, quad)
    }
    #[inline(always)]
    fn logup_p_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], &mut[alpha]: &mut[E; 1], i: usize) -> E {
        let operation = {
            let tmp = trace3[i].mul_into(&trace4[i]);
            trace2[i].mul_into(&trace1[i].add_into(&trace4[i].mul_into(&alpha))).add_into(&tmp)
        };
        // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
        operation
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn logup_fold_operation<E: FEXT>([trace1, trace2, trace3, trace4]: &mut [&mut [E]; 4], r: E, (ie, io): (usize, usize)) {
        // NB: even/low odd/high depending on index and folding order
        let (lowA, subA) = (trace1[ie], trace1[io]);
        let (lowB, subB) = (trace2[ie], trace2[io]);
        let (lowC, subC) = (trace3[ie], trace3[io]);
        let (lowD, subD) = (trace4[ie], trace4[io]);
        let foldA = lowA.add_into(&subA.mul_into(&r));
        let foldB = lowB.add_into(&subB.mul_into(&r));
        let foldC = lowC.add_into(&subC.mul_into(&r));
        let foldD = lowD.add_into(&subD.mul_into(&r));
        trace1[ie] = foldA;
        trace2[ie] = foldB;
        trace3[ie] = foldC;
        trace4[ie] = foldD;
    }
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let (mut multiacc1_raw, mut multiacc2_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace1, &trace2);
    let multiacc1: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc1_raw, trace1);
    let multiacc2: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc2_raw, trace2);
    tx.send(&multiacc1[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc2[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out = multiacc1[SKIPLAYERS].iter().zip(multiacc2[SKIPLAYERS].iter()).fold((E::ZERO,E::ONE), |(accnum, accden), (&num, &den)| {
            (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        });
        let info = format!("GKR OUTPUT: {out:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i>= M-SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim1, mut claim2) = {
        let z: &[E; SKIPLAYERS] = z.rsplit_array_ref::<SKIPLAYERS>().1;
        // (randeval_lowtohigh(multiacc1[SKIPLAYERS], z), randeval_lowtohigh(multiacc2[SKIPLAYERS], z))
        (randeval_hightolow(multiacc1[SKIPLAYERS], z), randeval_hightolow(multiacc2[SKIPLAYERS], z))
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc1: &mut[E] = multiacc1[i+1].as_mut();
        let acc2: &mut[E] = multiacc2[i+1].as_mut();
        let accz: &mut [E; i] = z.rsplit_array_mut::<i>().1;
        if DEBUG {
            let len = acc1.len();
            debug_assert!(len == acc2.len());
            println!("> gotta prove layer of len={len}: {acc1:?} and {acc2:?}");
            dbg!(&accz);
        }
        let mut acc1_evens = acc1.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc1_odds = acc1.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc2_evens = acc2.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc2_odds = acc2.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let alpha = tx.rand();
        let claim = claim1.add_into(&claim2.mul_into(&alpha));
        let traces = [&mut acc1_evens[..], &mut acc2_odds[..], &mut acc1_odds[..], &mut acc2_evens[..]];
        sum_eqspliteqKERNEL_lowtohigh::<_, _, _, _, _, _, _, 0>(tx, logup_g_operation, logup_p_operation, logup_fold_operation, traces, [alpha], accz, claim);
        // sum_eqspliteqKERNEL_hightolow::<_, _, _, _, _, _, 0>(tx, logup_g_operation, logup_p_operation, traces, [alpha], accz, claim);
        // 2to1
        let mut recv = [acc1_evens[0], acc1_odds[0], acc2_evens[0], acc2_odds[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim1 = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim2 = randeval_lowtohigh(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim1 = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim2 = randeval_hightolow(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[M-i-1] = r; // update point
    }});
}

fn gkr_grandsum_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let (mut multiacc1_raw, mut multiacc2_raw) = gen_multiaccsumtable_hightolow::<M, _, SKIPLAYERS>(&trace1, &trace2);
    let multiacc1: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc1_raw, trace1);
    let multiacc2: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc2_raw, trace2);
    tx.send(&multiacc1[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc2[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out = multiacc1[SKIPLAYERS].iter().zip(multiacc2[SKIPLAYERS].iter()).fold((E::ZERO,E::ONE), |(accnum, accden), (&num, &den)| {
            (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        });
        let info = format!("GKR OUTPUT: {out:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i < SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim1, mut claim2) = {
        let z: &[E; SKIPLAYERS] = z.split_array_ref::<SKIPLAYERS>().0;
        // (randeval_lowtohigh(multiacc1[SKIPLAYERS], z), randeval_lowtohigh(multiacc2[SKIPLAYERS], z))
        (randeval_hightolow(multiacc1[SKIPLAYERS], z), randeval_hightolow(multiacc2[SKIPLAYERS], z))
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc1: &mut[E] = multiacc1[i+1].as_mut();
        let acc2: &mut[E] = multiacc2[i+1].as_mut();
        let accz: &mut [E; i] = z.split_array_mut::<i>().0;
        if DEBUG {
            let len = acc1.len();
            debug_assert!(len == acc2.len());
            println!("> gotta prove layer of len={len}: {acc1:?} and {acc2:?}");
            dbg!(&accz);
        }
        let (acc1_lows, acc1_highs) = acc1.split_at_mut(1<<i);
        let (acc2_lows, acc2_highs) = acc2.split_at_mut(1<<i);
        let alpha = tx.rand();
        let claim = claim1.add_into(&claim2.mul_into(&alpha));
        // sum_eqspliteqlogup_lowtohigh::<_, _, 0>(tx, acc1_lows, acc2_highs, acc1_highs, acc2_lows, alpha, accz, claim);
        sum_eqspliteqlogup_hightolow::<_, _, 0>(tx, acc1_lows, acc2_highs, acc1_highs, acc2_lows, alpha, accz, claim);
        // 2to1
        let mut recv = [acc1_lows[0], acc1_highs[0], acc2_lows[0], acc2_highs[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim1 = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim2 = randeval_lowtohigh(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim1 = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim2 = randeval_hightolow(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[i] = r; // update point
    }});
}
fn gkr_grandsum_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let (mut multiacc1_raw, mut multiacc2_raw) = gen_multiaccsumtable_lowtohigh::<M, _, SKIPLAYERS>(&trace1, &trace2);
    let multiacc1: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc1_raw, trace1);
    let multiacc2: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc2_raw, trace2);
    tx.send(&multiacc1[SKIPLAYERS]); // this is all the verifier gets
    tx.send(&multiacc2[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out = multiacc1[SKIPLAYERS].iter().zip(multiacc2[SKIPLAYERS].iter()).fold((E::ZERO,E::ONE), |(accnum, accden), (&num, &den)| {
            (accnum.mul_into(&den).add_into(&num.mul_into(&accden)), accden.mul_into(&den))
        });
        let info = format!("GKR OUTPUT: {out:?}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i>= M-SKIPLAYERS {tx.rand()} else {E::ZERO});
    let (mut claim1, mut claim2) = {
        let z: &[E; SKIPLAYERS] = z.rsplit_array_ref::<SKIPLAYERS>().1;
        // (randeval_lowtohigh(multiacc1[SKIPLAYERS], z), randeval_lowtohigh(multiacc2[SKIPLAYERS], z))
        (randeval_hightolow(multiacc1[SKIPLAYERS], z), randeval_hightolow(multiacc2[SKIPLAYERS], z))
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc1: &mut[E] = multiacc1[i+1].as_mut();
        let acc2: &mut[E] = multiacc2[i+1].as_mut();
        let accz: &mut [E; i] = z.rsplit_array_mut::<i>().1;
        if DEBUG {
            let len = acc1.len();
            debug_assert!(len == acc2.len());
            println!("> gotta prove layer of len={len}: {acc1:?} and {acc2:?}");
            dbg!(&accz);
        }
        let mut acc1_evens = acc1.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc1_odds = acc1.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let mut acc2_evens = acc2.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc2_odds = acc2.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        let alpha = tx.rand();
        let claim = claim1.add_into(&claim2.mul_into(&alpha));
        sum_eqspliteqlogup_lowtohigh::<_, _, 0>(tx, &mut acc1_evens, &mut acc2_odds, &mut acc1_odds, &mut acc2_evens, alpha, accz, claim);
        // sum_eqspliteqlogup_hightolow::<_, _, 0>(tx, &mut acc1_evens, &mut acc2_odds, &mut acc1_odds, &mut acc2_evens, alpha, accz, claim);
        // 2to1
        let mut recv = [acc1_evens[0], acc1_odds[0], acc2_evens[0], acc2_odds[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim1 = randeval_lowtohigh(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        // claim2 = randeval_lowtohigh(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim1 = randeval_hightolow(&mut recv[..2], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim2 = randeval_hightolow(&mut recv[2..], &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[M-i-1] = r; // update point
    }});
}
fn gkr_grandproduct_hightolow<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace.len() == 1<<M);
    let mut multiacc_raw = gen_multiaccprodtable_hightolow::<M, _, SKIPLAYERS>(&trace);
    let multiacc: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_raw, trace);
    tx.send(&multiacc[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out =  multiacc[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let info = format!("GKR OUTPUT: {out}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i < SKIPLAYERS {tx.rand()} else {E::ZERO});
    let mut claim = {
        let z: &[E; SKIPLAYERS] = z.split_array_ref::<SKIPLAYERS>().0;
        // randeval_lowtohigh(multiacc[SKIPLAYERS], z)
        randeval_hightolow(multiacc[SKIPLAYERS], z)
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc: &mut[E] = multiacc[i+1].as_mut();
        let accz: &mut [E; i] = z.split_array_mut::<i>().0;
        if DEBUG {
            let len = acc.len();
            println!("> gotta prove layer of len={len}: {acc:?}");
            dbg!(&accz);
        }
        let (acc_lows, acc_highs) = acc.split_at_mut(1<<i);
        // sum_eqspliteqAB_lowtohigh::<_, _, 0>(tx, acc_lows, acc_highs, accz, claim);
        sum_eqspliteqAB_hightolow::<_, _, 0>(tx, acc_lows, acc_highs, accz, claim);
        // 2to1
        let mut recv = [acc_lows[0], acc_highs[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim = randeval_lowtohigh(&mut recv, &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim = randeval_hightolow(&mut recv, &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[i] = r; // update point
    }});
}
fn gkr_grandproduct_lowtohigh<const M: usize, E: FEXT, const SKIPLAYERS: usize>(tx: &mut Transcript<E>, trace: &mut [E]) // -> u8
where [(); M-SKIPLAYERS]:, [(); M+1]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(SKIPLAYERS < M && SKIPLAYERS >= 2);
    assert!(trace.len() == 1<<M);
    let mut multiacc_raw = gen_multiaccprodtable_lowtohigh::<M, _, SKIPLAYERS>(&trace);
    let multiacc: [&mut [E]; M+1] = slicemut_multiacctable::<M, _>(&mut multiacc_raw, trace);
    tx.send(&multiacc[SKIPLAYERS]); // this is all the verifier gets
    if DEBUG {
        let out =  multiacc[SKIPLAYERS].iter().fold(E::ONE, |acc, x| acc.mul_into(x));
        let info = format!("GKR OUTPUT: {out}"); dbg!(info);
    }
    let mut z: [E; M] = core::array::from_fn(|i| if i>= M-SKIPLAYERS {tx.rand()} else {E::ZERO});
    let mut claim = {
        let z: &[E; SKIPLAYERS] = z.rsplit_array_ref::<SKIPLAYERS>().1;
        // randeval_lowtohigh(multiacc[SKIPLAYERS], z)
        randeval_hightolow(multiacc[SKIPLAYERS], z)
    };
    seq_macro::seq!(i in 2..50 { if (SKIPLAYERS..M).contains(&i) {
        let acc: &mut[E] = multiacc[i+1].as_mut();
        let accz: &mut [E; i] = z.rsplit_array_mut::<i>().1;
        if DEBUG {
            let len = acc.len();
            println!("> gotta prove layer of len={len}: {acc:?}");
            dbg!(&accz);
        }
        let mut acc_evens = acc.iter().step_by(2).copied().collect::<Vec<E>>();
        let mut acc_odds = acc.iter().skip(1).step_by(2).copied().collect::<Vec<E>>();
        sum_eqspliteqAB_lowtohigh::<_, _, 0>(tx, &mut acc_evens, &mut acc_odds, accz, claim);
        // sum_eqspliteqAB_hightolow::<_, _, 0>(tx, &mut acc_evens, &mut acc_odds, accz, claim);
        // 2to1
        let mut recv = [acc_evens[0], acc_odds[0]];
        tx.send(&recv);
        let r = tx.rand();
        // claim = randeval_lowtohigh(&mut recv, &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        claim = randeval_hightolow(&mut recv, &[r]); // f(Xz) = f(0z) + X[f(1z)-f(0z)]
        z[M-i-1] = r; // update point
    }});
}

#[allow(non_snake_case)]
// TODO: find the algebraic error
fn sum_eqspliteqfull_hightolow<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace_readset_lows: &mut[E], trace_readset_highs: &mut[E], trace_writeset_lows: &mut[E], trace_writeset_highs: &mut[E], trace_lookupnum_lows: &mut[E], trace_lookupnum_highs: &mut[E], trace_lookupden_lows: &mut[E], trace_lookupden_highs: &mut[E], trace_rangechecknum_lows: &mut[E], trace_rangechecknum_highs: &mut[E], trace_rangecheckden_lows: &mut[E], trace_rangecheckden_highs: &mut[E], trace_timechecknum_lows: &mut[E], trace_timechecknum_highs: &mut[E], trace_timecheckden_lows: &mut[E], trace_timecheckden_highs: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {M/2-1+1}]:, [(); {M-M/2+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_glinquad_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, (ie, io): (usize, usize)) -> (E, E) {
    // NB: even/low odd/high depending on index and folding order
        #[inline(always)]
        #[allow(non_snake_case)]
        fn product_glinquad_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], (ie, io):(usize, usize)) -> (E, E) {
            // A*B
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            trace1[io] = subA;
            trace2[io] = subB;
            let lin = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
            let quad = subA.mul_into(&subB);
            (lin, quad)
        }
        #[inline(always)]
        #[allow(non_snake_case)]
        fn logup_glinquad_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut[E], alpha: E, (ie, io):(usize, usize)) -> (E, E) {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let (evenC, oddC) = (trace3[ie], trace3[io]);
            let (evenD, oddD) = (trace4[ie], trace4[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            let subC = oddC.sub_into(&evenC);
            let subD = oddD.sub_into(&evenD);
            trace1[io] = subA;
            trace2[io] = subB;
            trace3[io] = subC;
            trace4[io] = subD;
            let evenRLC = evenA.add_into(&evenB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let subRLC = subA.add_into(&subB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let lin = evenD.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subD)).add_into(&evenB.mul_into(&subC)).add_into(&evenC.mul_into(&subB));
            let quad = subD.mul_into(&subRLC).add_into(&subB.mul_into(&subC));
            (lin, quad)
        }
        let (lin_readset, quad_readset) = product_glinquad_operation(trace_readset0, trace_readset1, (ie, io));
        let (lin_writeset, quad_writeset) = product_glinquad_operation(trace_writeset0, trace_writeset1, (ie, io));
        let (lin_lookup, quad_lookup) = logup_glinquad_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, alpha1, (ie, io));
        let (lin_rangecheck, quad_rangecheck) = logup_glinquad_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, alpha2, (ie, io));
        let (lin_timecheck, quad_timecheck) = logup_glinquad_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, alpha3, (ie, io));
        let lin = lin_readset.add_into(&lin_writeset.mul_into(&alpha4)).add_into(&lin_lookup.mul_into(&alpha5)).add_into(&lin_rangecheck.mul_into(&alpha6)).add_into(&lin_timecheck.mul_into(&alpha7));
        let quad = quad_readset.add_into(&quad_writeset.mul_into(&alpha4)).add_into(&quad_lookup.mul_into(&alpha5)).add_into(&quad_rangecheck.mul_into(&alpha6)).add_into(&quad_timecheck.mul_into(&alpha7));
        (lin, quad)
    }
    #[inline(always)]
    fn full_point_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, i: usize) -> E {
        fn product_point_operation<E: FEXT>(trace1: &mut[E], trace2: &mut [E], i: usize) -> E {
            // A*B
            trace1[i].mul_into(&trace2[i])
        }
        fn logup_point_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut[E], alpha: E, i: usize) -> E {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let tmp = trace3[i].mul_into(&trace2[i]);
            trace4[i].mul_into(&trace1[i].add_into(&trace2[i].mul_into(&alpha))).add_into(&tmp)
        }
        let point_readset = product_point_operation(trace_readset0, trace_readset1, i);
        let point_writeset = product_point_operation(trace_writeset0, trace_writeset1, i);
        let point_lookup = logup_point_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, alpha1, i);
        let point_rangecheck = logup_point_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, alpha2, i);
        let point_timecheck = logup_point_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, alpha3, i);
        let point = point_readset.add_into(&point_writeset.mul_into(&alpha4)).add_into(&point_lookup.mul_into(&alpha5)).add_into(&point_rangecheck.mul_into(&alpha6)).add_into(&point_timecheck.mul_into(&alpha7));
        point
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_fold_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], r: E, (ie, io): (usize, usize)) {
    // NB: even/low odd/high depending on index and folding order
        fn product_fold_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        fn logup_fold_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut [E], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        product_fold_operation(trace_readset0, trace_readset1, r, (ie, io));
        product_fold_operation(trace_writeset0, trace_writeset1, r, (ie, io));
        logup_fold_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, r, (ie, io));
        logup_fold_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, r, (ie, io));
        logup_fold_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, r, (ie, io));
    }

    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(trace_readset_lows.len() == 1<<M && trace_writeset_lows.len() == 1<<M && trace_lookupnum_lows.len() == 1<<M && trace_lookupden_lows.len() == 1<<M);
    assert!(trace_rangechecknum_lows.len() == 1<<M && trace_rangecheckden_lows.len() == 1<<M && trace_timechecknum_lows.len() == 1<<M && trace_timecheckden_lows.len() == 1<<M);
    assert!(trace_readset_highs.len() == 1<<M && trace_writeset_highs.len() == 1<<M && trace_lookupnum_highs.len() == 1<<M && trace_lookupden_highs.len() == 1<<M);
    assert!(trace_rangechecknum_highs.len() == 1<<M && trace_rangecheckden_highs.len() == 1<<M && trace_timechecknum_highs.len() == 1<<M && trace_timecheckden_highs.len() == 1<<M);
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_lowtohigh::<{M/2-1}, _>(z[M2..M-1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_lowtohigh::<{M-M/2}, _>(z[..M2].try_into().unwrap());
    let multieq1 = slicerev_multieqtable::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace_readset_lows, &trace_writeset_lows, &trace_lookupnum_lows, &trace_lookupden_lows, &trace_rangechecknum_lows, &trace_rangecheckden_lows, &trace_timechecknum_lows, &trace_timecheckden_lows);
        dbg!(&trace_readset_highs, &trace_writeset_highs, &trace_lookupnum_highs, &trace_lookupden_highs, &trace_rangechecknum_highs, &trace_rangecheckden_highs, &trace_timechecknum_highs, &trace_timecheckden_highs);
        dbg!(&z);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part2,x_part1) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part2, x_part1)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<M-round-1;
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xM-i, zM-i)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie2 = i2;
                    let io2 = ie2 + step;
                    let ie = ie2 | (i1<<M2);
                    let io = io2 | (i1<<M2);
                    let (lin, quad) = full_glinquad_operation(trace_readset_lows, trace_readset_highs, trace_writeset_lows, trace_writeset_highs, trace_lookupnum_lows, trace_lookupnum_highs, trace_lookupden_lows, trace_lookupden_highs, trace_rangechecknum_lows, trace_rangechecknum_highs, trace_rangecheckden_lows, trace_rangecheckden_highs, trace_timechecknum_lows, trace_timechecknum_highs, trace_timecheckden_lows, trace_timecheckden_highs, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, (ie, io));
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M-1-round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M-1-round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M-1-round] = r;
        // fold all Traces
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            full_fold_operation(trace_readset_lows, trace_readset_highs, trace_writeset_lows, trace_writeset_highs, trace_lookupnum_lows, trace_lookupnum_highs, trace_lookupden_lows, trace_lookupden_highs, trace_rangechecknum_lows, trace_rangechecknum_highs, trace_rangecheckden_lows, trace_rangecheckden_highs, trace_timechecknum_lows, trace_timechecknum_highs, trace_timecheckden_lows, trace_timecheckden_highs, r, (ie, io));
        }
        if DEBUG {
            let traces = [&mut trace_readset_lows[..], &mut trace_readset_highs[..], &mut trace_writeset_lows[..], &mut trace_writeset_highs[..], &mut trace_lookupnum_lows[..], &mut trace_lookupnum_highs[..], &mut trace_lookupden_lows[..], &mut trace_lookupden_highs[..], &mut trace_rangechecknum_lows[..], &mut trace_rangechecknum_highs[..], &mut trace_rangecheckden_lows[..], &mut trace_rangecheckden_highs[..], &mut trace_timechecknum_lows[..], &mut trace_timechecknum_highs[..], &mut trace_timecheckden_lows[..], &mut trace_timecheckden_highs[..]];
            let layer_parts = traces.into_iter().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x_part2, x) EQ2(x_part2) EQ1(x) OP(x_part2, x)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    debug_assert!(M-(M1-1) == M2+1);
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<M2-round;
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2;
                let io2 = ie2 + step;
                let ie = ie2;
                let io = io2;
                let (lin, quad) = full_glinquad_operation(trace_readset_lows, trace_readset_highs, trace_writeset_lows, trace_writeset_highs, trace_lookupnum_lows, trace_lookupnum_highs, trace_lookupden_lows, trace_lookupden_highs, trace_rangechecknum_lows, trace_rangechecknum_highs, trace_rangecheckden_lows, trace_rangecheckden_highs, trace_timechecknum_lows, trace_timechecknum_highs, trace_timecheckden_lows, trace_timecheckden_highs, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, (ie, io));
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M2-round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M2-round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M2-round] = r;
        // fold all Traces
        for i2 in 0..1<<M2-round {
            let ie2 = i2;
            let io2 = ie2 + step;
            let ie = ie2;
            let io = io2;
            full_fold_operation(trace_readset_lows, trace_readset_highs, trace_writeset_lows, trace_writeset_highs, trace_lookupnum_lows, trace_lookupnum_highs, trace_lookupden_lows, trace_lookupden_highs, trace_rangechecknum_lows, trace_rangechecknum_highs, trace_rangecheckden_lows, trace_rangecheckden_highs, trace_timechecknum_lows, trace_timechecknum_highs, trace_timecheckden_lows, trace_timecheckden_highs, r, (ie, io));
        }
        if DEBUG {
            let traces = [&mut trace_readset_lows[..], &mut trace_readset_highs[..], &mut trace_writeset_lows[..], &mut trace_writeset_highs[..], &mut trace_lookupnum_lows[..], &mut trace_lookupnum_highs[..], &mut trace_lookupden_lows[..], &mut trace_lookupden_highs[..], &mut trace_rangechecknum_lows[..], &mut trace_rangechecknum_highs[..], &mut trace_rangecheckden_lows[..], &mut trace_rangecheckden_highs[..], &mut trace_timechecknum_lows[..], &mut trace_timechecknum_highs[..], &mut trace_timecheckden_lows[..], &mut trace_timecheckden_highs[..]];
            let layer_parts = traces.into_iter().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // // explicit sum
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        // TODO: is it not better to just repeatedly fold?
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[..SKIPVARS].try_into().unwrap());
        let mut sum = E::ZERO;
        for i in 0..1<<SKIPVARS {
            let operation = full_point_operation(trace_readset_lows, trace_readset_highs, trace_writeset_lows, trace_writeset_highs, trace_lookupnum_lows, trace_lookupnum_highs, trace_lookupden_lows, trace_lookupden_highs, trace_rangechecknum_lows, trace_rangechecknum_highs, trace_rangecheckden_lows, trace_rangecheckden_highs, trace_timechecknum_lows, trace_timechecknum_highs, trace_timecheckden_lows, trace_timecheckden_highs, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, i);
            sum = sum.add_into(&operation.mul_into(&eq_final[i]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}
#[allow(non_snake_case)]
fn sum_eqspliteqfull_lowtohigh<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace_readset_evens: &mut[E], trace_readset_odds: &mut[E], trace_writeset_evens: &mut[E], trace_writeset_odds: &mut[E], trace_lookupnum_evens: &mut[E], trace_lookupnum_odds: &mut[E], trace_lookupden_evens: &mut[E], trace_lookupden_odds: &mut[E], trace_rangechecknum_evens: &mut[E], trace_rangechecknum_odds: &mut[E], trace_rangecheckden_evens: &mut[E], trace_rangecheckden_odds: &mut[E], trace_timechecknum_evens: &mut[E], trace_timechecknum_odds: &mut[E], trace_timecheckden_evens: &mut[E], trace_timecheckden_odds: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {(M/2-1)+1}]:, [(); {(M-M/2)+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_glinquad_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, (ie, io): (usize, usize)) -> (E, E) {
    // NB: even/low odd/high depending on index and folding order
        #[inline(always)]
        #[allow(non_snake_case)]
        fn product_glinquad_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], (ie, io):(usize, usize)) -> (E, E) {
            // A*B
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            trace1[io] = subA;
            trace2[io] = subB;
            let lin = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
            let quad = subA.mul_into(&subB);
            (lin, quad)
        }
        #[inline(always)]
        #[allow(non_snake_case)]
        fn logup_glinquad_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut[E], alpha: E, (ie, io):(usize, usize)) -> (E, E) {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let (evenA, oddA) = (trace1[ie], trace1[io]);
            let (evenB, oddB) = (trace2[ie], trace2[io]);
            let (evenC, oddC) = (trace3[ie], trace3[io]);
            let (evenD, oddD) = (trace4[ie], trace4[io]);
            let subA = oddA.sub_into(&evenA);
            let subB = oddB.sub_into(&evenB);
            let subC = oddC.sub_into(&evenC);
            let subD = oddD.sub_into(&evenD);
            trace1[io] = subA;
            trace2[io] = subB;
            trace3[io] = subC;
            trace4[io] = subD;
            let evenRLC = evenA.add_into(&evenB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let subRLC = subA.add_into(&subB.clone().mul_into(&alpha)); // TODO: remove clone with preproc.
            let lin = evenD.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subD)).add_into(&evenB.mul_into(&subC)).add_into(&evenC.mul_into(&subB));
            let quad = subD.mul_into(&subRLC).add_into(&subB.mul_into(&subC));
            (lin, quad)
        }
        let (lin_readset, quad_readset) = product_glinquad_operation(trace_readset0, trace_readset1, (ie, io));
        let (lin_writeset, quad_writeset) = product_glinquad_operation(trace_writeset0, trace_writeset1, (ie, io));
        let (lin_lookup, quad_lookup) = logup_glinquad_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, alpha1, (ie, io));
        let (lin_rangecheck, quad_rangecheck) = logup_glinquad_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, alpha2, (ie, io));
        let (lin_timecheck, quad_timecheck) = logup_glinquad_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, alpha3, (ie, io));
        let lin = lin_readset.add_into(&lin_writeset.mul_into(&alpha4)).add_into(&lin_lookup.mul_into(&alpha5)).add_into(&lin_rangecheck.mul_into(&alpha6)).add_into(&lin_timecheck.mul_into(&alpha7));
        let quad = quad_readset.add_into(&quad_writeset.mul_into(&alpha4)).add_into(&quad_lookup.mul_into(&alpha5)).add_into(&quad_rangecheck.mul_into(&alpha6)).add_into(&quad_timecheck.mul_into(&alpha7));
        (lin, quad)
    }
    #[inline(always)]
    fn full_point_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], alpha1: E, alpha2: E, alpha3: E, alpha4: E, alpha5: E, alpha6: E, alpha7: E, i: usize) -> E {
        fn product_point_operation<E: FEXT>(trace1: &mut[E], trace2: &mut [E], i: usize) -> E {
            // A*B
            trace1[i].mul_into(&trace2[i])
        }
        fn logup_point_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut[E], alpha: E, i: usize) -> E {
            // A/B + C/D = (AD + BC)/BD = AD + BC + alphaBD = D(A + alphaB) + BC
            let tmp = trace3[i].mul_into(&trace2[i]);
            trace4[i].mul_into(&trace1[i].add_into(&trace2[i].mul_into(&alpha))).add_into(&tmp)
        }
        let point_readset = product_point_operation(trace_readset0, trace_readset1, i);
        let point_writeset = product_point_operation(trace_writeset0, trace_writeset1, i);
        let point_lookup = logup_point_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, alpha1, i);
        let point_rangecheck = logup_point_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, alpha2, i);
        let point_timecheck = logup_point_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, alpha3, i);
        let point = point_readset.add_into(&point_writeset.mul_into(&alpha4)).add_into(&point_lookup.mul_into(&alpha5)).add_into(&point_rangecheck.mul_into(&alpha6)).add_into(&point_timecheck.mul_into(&alpha7));
        point
    }
    #[inline(always)]
    #[allow(non_snake_case)]
    fn full_fold_operation<E: FEXT>(trace_readset0: &mut[E], trace_readset1: &mut[E], trace_writeset0: &mut[E], trace_writeset1: &mut[E], trace_lookupnum0: &mut[E], trace_lookupnum1: &mut[E], trace_lookupden0: &mut[E], trace_lookupden1: &mut[E], trace_rangechecknum0: &mut[E], trace_rangechecknum1: &mut[E], trace_rangecheckden0: &mut[E], trace_rangecheckden1: &mut[E], trace_timechecknum0: &mut[E], trace_timechecknum1: &mut[E], trace_timecheckden0: &mut[E], trace_timecheckden1: &mut[E], r: E, (ie, io): (usize, usize)) {
    // NB: even/low odd/high depending on index and folding order
        fn product_fold_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        fn logup_fold_operation<E: FEXT>(trace1: &mut[E], trace2: &mut[E], trace3: &mut[E], trace4: &mut [E], r: E, (ie, io): (usize, usize)) {
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        product_fold_operation(trace_readset0, trace_readset1, r, (ie, io));
        product_fold_operation(trace_writeset0, trace_writeset1, r, (ie, io));
        logup_fold_operation(trace_lookupnum0, trace_lookupden0, trace_lookupnum1, trace_lookupden1, r, (ie, io));
        logup_fold_operation(trace_rangechecknum0, trace_rangecheckden0, trace_rangechecknum1, trace_rangecheckden1, r, (ie, io));
        logup_fold_operation(trace_timechecknum0, trace_timecheckden0, trace_timechecknum1, trace_timecheckden1, r, (ie, io));
    }
    
    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(trace_readset_evens.len() == 1<<M && trace_writeset_evens.len() == 1<<M && trace_lookupnum_evens.len() == 1<<M && trace_lookupden_evens.len() == 1<<M);
    assert!(trace_rangechecknum_evens.len() == 1<<M && trace_rangecheckden_evens.len() == 1<<M && trace_timechecknum_evens.len() == 1<<M && trace_timecheckden_evens.len() == 1<<M);
    assert!(trace_readset_odds.len() == 1<<M && trace_writeset_odds.len() == 1<<M && trace_lookupnum_odds.len() == 1<<M && trace_lookupden_odds.len() == 1<<M);
    assert!(trace_rangechecknum_odds.len() == 1<<M && trace_rangecheckden_odds.len() == 1<<M && trace_timechecknum_odds.len() == 1<<M && trace_timecheckden_odds.len() == 1<<M);
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M/2-1}, _>(z[1..M1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M-M/2}, _>(z[M1..].try_into().unwrap());
    let multieq1 = slicerev_multieqtable_antiscattered::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable_antiscattered::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace_readset_evens, &trace_writeset_evens, &trace_lookupnum_evens, &trace_lookupden_evens, &trace_rangechecknum_evens, &trace_rangecheckden_evens, &trace_timechecknum_evens, &trace_timecheckden_evens);
        dbg!(&trace_readset_odds, &trace_writeset_odds, &trace_lookupnum_odds, &trace_lookupden_odds, &trace_rangechecknum_odds, &trace_rangecheckden_odds, &trace_timechecknum_odds, &trace_timecheckden_odds);
        dbg!(&z);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part1,x_part2) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part1, x_part2)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<round;
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie1 = i1*step*2;
                    let io1 = ie1 + step;
                    let ie = ie1 | (i2<<M1);
                    let io = io1 | (i2<<M1);
                    let (lin, quad) = full_glinquad_operation(trace_readset_evens, trace_readset_odds, trace_writeset_evens, trace_writeset_odds, trace_lookupnum_evens, trace_lookupnum_odds, trace_lookupden_evens, trace_lookupden_odds, trace_rangechecknum_evens, trace_rangechecknum_odds, trace_rangecheckden_evens, trace_rangecheckden_odds, trace_timechecknum_evens, trace_timechecknum_odds, trace_timecheckden_evens, trace_timecheckden_odds, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, (ie, io));
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[round] = r; // update z in place
        // fold all Traces
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            full_fold_operation(trace_readset_evens, trace_readset_odds, trace_writeset_evens, trace_writeset_odds, trace_lookupnum_evens, trace_lookupnum_odds, trace_lookupden_evens, trace_lookupden_odds, trace_rangechecknum_evens, trace_rangechecknum_odds, trace_rangecheckden_evens, trace_rangecheckden_odds, trace_timechecknum_evens, trace_timechecknum_odds, trace_timecheckden_evens, trace_timecheckden_odds, r, (ie, io));
        }
        if DEBUG {
            let traces = [&mut trace_readset_evens[..], &mut trace_readset_odds[..], &mut trace_writeset_evens[..], &mut trace_writeset_odds[..], &mut trace_lookupnum_evens[..], &mut trace_lookupnum_odds[..], &mut trace_lookupden_evens[..], &mut trace_lookupden_odds[..], &mut trace_rangechecknum_evens[..], &mut trace_rangechecknum_odds[..], &mut trace_rangecheckden_evens[..], &mut trace_rangecheckden_odds[..], &mut trace_timechecknum_evens[..], &mut trace_timechecknum_odds[..], &mut trace_timecheckden_evens[..], &mut trace_timecheckden_odds[..]];
            let layer_parts = traces.into_iter().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x, x_part2) EQ2(x_part2) EQ1(x) OP(x, x_part2)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<round;
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2*step*2;
                let io2 = ie2 + step;
                let ie = ie2<<M1-1;
                let io = io2<<M1-1;
                let (lin, quad) = full_glinquad_operation(trace_readset_evens, trace_readset_odds, trace_writeset_evens, trace_writeset_odds, trace_lookupnum_evens, trace_lookupnum_odds, trace_lookupden_evens, trace_lookupden_odds, trace_rangechecknum_evens, trace_rangechecknum_odds, trace_rangecheckden_evens, trace_rangecheckden_odds, trace_timechecknum_evens, trace_timechecknum_odds, trace_timecheckden_evens, trace_timecheckden_odds, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, (ie, io));
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M1-1+round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M1-1+round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M1-1+round] = r; // update z with folding randomness
        // fold all Traces
        for i2 in 0..1<<M2-round {
            let ie2 = i2*step*2;
            let io2 = ie2 + step;
            let ie = ie2<<M1-1;
            let io = io2<<M1-1;
            full_fold_operation(trace_readset_evens, trace_readset_odds, trace_writeset_evens, trace_writeset_odds, trace_lookupnum_evens, trace_lookupnum_odds, trace_lookupden_evens, trace_lookupden_odds, trace_rangechecknum_evens, trace_rangechecknum_odds, trace_rangecheckden_evens, trace_rangecheckden_odds, trace_timechecknum_evens, trace_timechecknum_odds, trace_timecheckden_evens, trace_timecheckden_odds, r, (ie, io));
        }
        if DEBUG {
            let traces = [&mut trace_readset_evens[..], &mut trace_readset_odds[..], &mut trace_writeset_evens[..], &mut trace_writeset_odds[..], &mut trace_lookupnum_evens[..], &mut trace_lookupnum_odds[..], &mut trace_lookupden_evens[..], &mut trace_lookupden_odds[..], &mut trace_rangechecknum_evens[..], &mut trace_rangechecknum_odds[..], &mut trace_rangecheckden_evens[..], &mut trace_rangecheckden_odds[..], &mut trace_timechecknum_evens[..], &mut trace_timechecknum_odds[..], &mut trace_timecheckden_evens[..], &mut trace_timecheckden_odds[..]];
            let layer_parts = traces.into_iter().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // // explicit sum
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        // TODO: is it not better to just repeatedly fold?
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[M-SKIPVARS..].try_into().unwrap());
        let step = 1<<M-SKIPVARS;
        let mut sum = E::ZERO;
        for i2 in 0..1<<SKIPVARS {
            let i = i2*step;
            let operation = full_point_operation(trace_readset_evens, trace_readset_odds, trace_writeset_evens, trace_writeset_odds, trace_lookupnum_evens, trace_lookupnum_odds, trace_lookupden_evens, trace_lookupden_odds, trace_rangechecknum_evens, trace_rangechecknum_odds, trace_rangecheckden_evens, trace_rangecheckden_odds, trace_timechecknum_evens, trace_timechecknum_odds, trace_timecheckden_evens, trace_timecheckden_odds, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7, i);
            sum = sum.add_into(&operation.mul_into(&eq_final[i2]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}
#[allow(non_snake_case)]
fn sum_eqspliteqKERNEL_hightolow<const M: usize, const T: usize, const A: usize, GFuncKernel, PointKernel, FoldKernel, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, g_operation: GFuncKernel, p_operation: PointKernel, fold_operation: FoldKernel, mut traces: [&mut[E]; T], mut alphas: [E; A], z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {M/2-1+1}]:, [(); {M-M/2+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
      GFuncKernel: Fn(&mut [&mut [E]; T], &mut[E; A], (usize, usize)) -> (E, E),
      PointKernel: Fn(&mut [&mut [E]; T], &mut[E; A], usize) -> E,
      FoldKernel: Fn(&mut [&mut [E]; T], E, (usize, usize)),
{
    // let trace1 = unsafe{ core::slice::from_raw_parts_mut(traces[0].as_mut_ptr(), 1<<M)};
    // let trace2 = unsafe{ core::slice::from_raw_parts_mut(traces[1].as_mut_ptr(), 1<<M)};
    // let trace3 = unsafe{ core::slice::from_raw_parts_mut(traces[2].as_mut_ptr(), 1<<M)};
    // let trace4 = unsafe{ core::slice::from_raw_parts_mut(traces[3].as_mut_ptr(), 1<<M)};
    // let [alpha] = alphas[..] else {panic!()};

    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(traces.iter().all(|trace| trace.len() == 1<<M));
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_lowtohigh::<{M/2-1}, _>(z[M2..M-1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_lowtohigh::<{M-M/2}, _>(z[..M2].try_into().unwrap());
    let multieq1 = slicerev_multieqtable::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&traces);
        dbg!(&z);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part2,x_part1) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part2, x_part1)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<M-round-1;
        // let z = z[M-1-round]; // be careful with override
        // dbg!(M-1-round);
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xM-i, zM-i)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie2 = i2;
                    let io2 = ie2 + step;
                    let ie = ie2 | (i1<<M2);
                    let io = io2 | (i1<<M2);
                    // let (lowA, highA) = (trace1[ie], trace1[io]);
                    // let (lowB, highB) = (trace2[ie], trace2[io]);
                    // let (lowC, highC) = (trace3[ie], trace3[io]);
                    // let (lowD, highD) = (trace4[ie], trace4[io]);
                    // let subA = (P - lowA + highA) % P;
                    // let subB = (P - lowB + highB) % P;
                    // let subC = (P - lowC + highC) % P;
                    // let subD = (P - lowD + highD) % P;
                    // trace1[io] = subA;
                    // trace2[io] = subB;
                    // trace3[io] = subC;
                    // trace4[io] = subD;
                    // let lowRLC = (lowA + alpha*lowD) % P;
                    // let subRLC = (subA + alpha*subD) % P;
                    // let lin = (lowB*subRLC + subB*lowRLC + lowC*subD + subC*lowD) % P;
                    // let quad = (subB*subRLC + subC*subD) % P;
                    let (lin, quad) = g_operation(&mut traces, &mut alphas, (ie, io)); // TODO: try without &mut
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!(":-)");
        // }
        let r = tx.rand();
        claim_innereval = {
            let z = z[M-1-round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M-1-round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M-1-round] = r;
        // fold all Traces
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            // let (lowA, subA) = (trace1[ie], trace1[io]);
            // let (lowB, subB) = (trace2[ie], trace2[io]);
            // let (lowC, subC) = (trace3[ie], trace3[io]);
            // let (lowD, subD) = (trace4[ie], trace4[io]);
            // let foldA = (lowA + r*subA) % P;
            // let foldB = (lowB + r*subB) % P;
            // let foldC = (lowC + r*subC) % P;
            // let foldD = (lowD + r*subD) % P;
            // trace1[ie] = foldA;
            // trace2[ie] = foldB;
            // trace3[ie] = foldC;
            // trace4[ie] = foldD;
            // for trace in &mut traces { // assumes Kernel precomputed the subs...
            //     let (lowT, subT) = (trace[ie], trace[io]);
            //     let foldT = (lowT + r*subT) % P;
            //     trace[ie] = foldT;
            // }
            fold_operation(&mut traces, r, (ie, io));
        }
        if DEBUG {
            // let layer_partA = &trace1[..1<<M-round-1];
            // let layer_partB = &trace2[..1<<M-round-1];
            // let layer_partC = &trace3[..1<<M-round-1];
            // let layer_partD = &trace4[..1<<M-round-1];
            let layer_parts = traces.iter_mut().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x_part2, x) EQ2(x_part2) EQ1(x) OP(x_part2, x)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    debug_assert!(M-(M1-1) == M2+1);
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<M2-round;
        // let z = z[M2-round];
        // dbg!(M2-round);
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2;
                let io2 = ie2 + step;
                let ie = ie2;
                let io = io2;
                // let (lowA,  highA) = (trace1[ie], trace1[io]);
                // let (lowB,  highB) = (trace2[ie], trace2[io]);
                // let (lowC,  highC) = (trace3[ie], trace3[io]);
                // let (lowD,  highD) = (trace4[ie], trace4[io]);
                // let subA = (P - lowA +  highA) % P;
                // let subB = (P - lowB +  highB) % P;
                // let subC = (P - lowC +  highC) % P;
                // let subD = (P - lowD +  highD) % P;
                // trace1[io] = subA;
                // trace2[io] = subB;
                // trace3[io] = subC;
                // trace4[io] = subD;
                // let lowRLC = (lowA + alpha*lowD) % P;
                // let subRLC = (subA + alpha*subD) % P;
                // let lin = (lowB*subRLC + subB*lowRLC + lowC*subD + subC*lowD) % P;
                // let quad = (subB*subRLC + subC*subD) % P;
                let (lin, quad) = g_operation(&mut traces, &mut alphas, (ie, io)); // TODO: try without &mut
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!("B-)");
        // }
        let r = tx.rand();
        claim_innereval = {
            let z = z[M2-round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M2-round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M2-round] = r;
        // fold all Traces
        for i2 in 0..1<<M2-round {
            let ie2 = i2;
            let io2 = ie2 + step;
            let ie = ie2;
            let io = io2;
            // let (lowA, subA) = (trace1[ie], trace1[io]);
            // let (lowB, subB) = (trace2[ie], trace2[io]);
            // let (lowC, subC) = (trace3[ie], trace3[io]);
            // let (lowD, subD) = (trace4[ie], trace4[io]);
            // let foldA = (lowA + r*subA) % P;
            // let foldB = (lowB + r*subB) % P;
            // let foldC = (lowC + r*subC) % P;
            // let foldD = (lowD + r*subD) % P;
            // trace1[ie] = foldA;
            // trace2[ie] = foldB;
            // trace3[ie] = foldC;
            // trace4[ie] = foldD;
            // for trace in &mut traces { // assumes Kernel precomputed the subs...
            //     let (lowT, subT) = (trace[ie], trace[io]);
            //     let foldT = (lowT + r*subT) % P;
            //     trace[ie] = foldT;
            // }
            fold_operation(&mut traces, r, (ie, io));
        }
        if DEBUG {
            // let layer_partA = &trace1[..1<<M2-round];
            // let layer_partB = &trace2[..1<<M2-round];
            // let layer_partC = &trace3[..1<<M2-round];
            // let layer_partD = &trace4[..1<<M2-round];
            let layer_parts = traces.iter_mut().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
            // dbg!(layer_partA.len());
        }
    }

    // let final_partA = &trace1[..1<<SKIPVARS];
    // let final_partB = &trace2[..1<<SKIPVARS];
    // send(&final_partA[..]);
    // send(&final_partB[..]);
    // // explicit sum
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(&final_partA, &final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[..SKIPVARS].try_into().unwrap());
        let mut sum = E::ZERO;
        for i in 0..1<<SKIPVARS {
            let operation = p_operation(&mut traces, &mut alphas, i);
            // let operation = trace2[i]*(trace1[i]+alpha*trace4[i]) + trace3[i]*trace4[i];
            // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
            sum = sum.add_into(&operation.mul_into(&eq_final[i]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}
#[allow(non_snake_case)]
fn sum_eqspliteqKERNEL_lowtohigh<const M: usize, const T: usize, const A: usize, GFuncKernel, PointKernel, FoldKernel, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, g_operation: GFuncKernel, p_operation: PointKernel, fold_operation: FoldKernel, mut traces: [&mut[E]; T], mut alphas: [E; A], z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {(M/2-1)+1}]:, [(); {(M-M/2)+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
      GFuncKernel: Fn(&mut [&mut [E]; T], &mut[E; A], (usize, usize)) -> (E, E),
      PointKernel: Fn(&mut [&mut [E]; T], &mut[E; A], usize) -> E,
      FoldKernel: Fn(&mut [&mut [E]; T], E, (usize, usize)),
{
    // let trace1 = unsafe{ core::slice::from_raw_parts_mut(traces[0].as_mut_ptr(), 1<<M)};
    // let trace2 = unsafe{ core::slice::from_raw_parts_mut(traces[1].as_mut_ptr(), 1<<M)};
    // let trace3 = unsafe{ core::slice::from_raw_parts_mut(traces[2].as_mut_ptr(), 1<<M)};
    // let trace4 = unsafe{ core::slice::from_raw_parts_mut(traces[3].as_mut_ptr(), 1<<M)};
    // let [alpha] = alphas[..] else {panic!()};


    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(traces.iter().all(|trace| trace.len() == 1<<M));
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M/2-1}, _>(z[1..M1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M-M/2}, _>(z[M1..].try_into().unwrap());
    let multieq1 = slicerev_multieqtable_antiscattered::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable_antiscattered::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&traces);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part1,x_part2) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part1, x_part2)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<round;
        // let z = z[round]; // be careful with override
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie1 = i1*step*2;
                    let io1 = ie1 + step;
                    let ie = ie1 | (i2<<M1);
                    let io = io1 | (i2<<M1);
                    // let (evenA, oddA) = (trace1[ie], trace1[io]);
                    // let (evenB, oddB) = (trace2[ie], trace2[io]);
                    // let (evenC, oddC) = (trace3[ie], trace3[io]);
                    // let (evenD, oddD) = (trace4[ie], trace4[io]);
                    // let subA = (P - evenA + oddA) % P;
                    // let subB = (P - evenB + oddB) % P;
                    // let subC = (P - evenC + oddC) % P;
                    // let subD = (P - evenD + oddD) % P;
                    // trace1[io] = subA;
                    // trace2[io] = subB;
                    // trace3[io] = subC;
                    // trace4[io] = subD;
                    // let evenRLC = (evenA + alpha*evenD) % P;
                    // let subRLC = (subA + alpha*subD) % P;
                    // let lin = (evenB*subRLC + subB*evenRLC + evenC*subD + subC*evenD) % P;
                    // let quad = (subB*subRLC + subC*subD) % P;
                    let (lin, quad) = g_operation(&mut traces, &mut alphas, (ie, io)); // TODO: try without &mut
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r = tx.rand();
        claim_innereval = {
            let z = z[round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[round] = r; // update z in place
        // fold all Traces
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            // for trace in &mut traces { // assumes Kernel precomputed the subs...
            //     let (evenT, subT) = (trace[ie], trace[io]);
            //     let foldT = (evenT + r*subT) % P;
            //     trace[ie] = foldT;
            // }
            fold_operation(&mut traces, r, (ie, io));
        }
        if DEBUG {
            let layer_parts = traces.iter_mut().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x, x_part2) EQ2(x_part2) EQ1(x) OP(x, x_part2)
    // with OP(x) from Kernel
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<round;
        // let z = z[M1-1+round]; // be careful with override
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2*step*2;
                let io2 = ie2 + step;
                let ie = ie2<<M1-1;
                let io = io2<<M1-1;
                // let (evenA, oddA) = (trace1[ie], trace1[io]);
                // let (evenB, oddB) = (trace2[ie], trace2[io]);
                // let (evenC, oddC) = (trace3[ie], trace3[io]);
                // let (evenD, oddD) = (trace4[ie], trace4[io]);
                // let subA = (P - evenA + oddA) % P;
                // let subB = (P - evenB + oddB) % P;
                // let subC = (P - evenC + oddC) % P;
                // let subD = (P - evenD + oddD) % P;
                // trace1[io] = subA;
                // trace2[io] = subB;
                // trace3[io] = subC;
                // trace4[io] = subD;
                // let evenRLC = (evenA + alpha*evenD) % P;
                // let subRLC = (subA + alpha*subD) % P;
                // let lin = (evenB*subRLC + subB*evenRLC + evenC*subD + subC*evenD) % P;
                // let quad = (subB*subRLC + subC*subD) % P;
                let (lin, quad) = g_operation(&mut traces, &mut alphas, (ie, io)); // TODO: try without &mut
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r = tx.rand();
        claim_innereval = {
            let z = z[M1-1+round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M1-1+round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M1-1+round] = r; // update z with folding randomness
        // fold all Traces
        for i2 in 0..1<<M2-round {
            let ie2 = i2*step*2;
            let io2 = ie2 + step;
            let ie = ie2<<M1-1;
            let io = io2<<M1-1;
            // let (evenA, subA) = (trace1[ie], trace1[io]);
            // let (evenB, subB) = (trace2[ie], trace2[io]);
            // let (evenC, subC) = (trace3[ie], trace3[io]);
            // let (evenD, subD) = (trace4[ie], trace4[io]);
            // let foldA = (evenA + r*subA) % P;
            // let foldB = (evenB + r*subB) % P;
            // let foldC = (evenC + r*subC) % P;
            // let foldD = (evenD + r*subD) % P;
            // trace1[ie] = foldA;
            // trace2[ie] = foldB;
            // trace3[ie] = foldC;
            // trace4[ie] = foldD;
            // for trace in &mut traces { // assumes Kernel precomputed the subs...
            //     let (evenT, subT) = (trace[ie], trace[io]);
            //     let foldT = (evenT + r*subT) % P;
            //     trace[ie] = foldT;
            // }
            fold_operation(&mut traces, r, (ie, io));
        }
        if DEBUG {
            // let layer_partA = trace1.iter_mut().step_by(step<<M1).collect::<Vec<&mut u8>>();
            // let layer_partB = trace2.iter_mut().step_by(step<<M1).collect::<Vec<&mut u8>>();
            // let layer_partC = trace3.iter_mut().step_by(step<<M1).collect::<Vec<&mut u8>>();
            // let layer_partD = trace4.iter_mut().step_by(step<<M1).collect::<Vec<&mut u8>>();
            let layer_parts = traces.iter_mut().map(|trace| trace.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>());
            dbg!(layer_parts, claim_eqscale, claim_innereval);
        }
    }

    // // explicit sum
    // let final_partA = trace1.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // let final_partB = trace2.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // send(final_partA.as_slice());
    // send(final_partB.as_slice());
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(final_partA, final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        // TODO: is it not better to just repeatedly fold?
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[M-SKIPVARS..].try_into().unwrap());
        let step = 1<<M-SKIPVARS;
        let mut sum = E::ZERO;
        for i2 in 0..1<<SKIPVARS {
            let i = i2*step;
            let operation = p_operation(&mut traces, &mut alphas, i);
            // let operation = trace2[i]*(trace1[i]+alpha*trace4[i]) + trace3[i]*trace4[i];
            // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
            sum = sum.add_into(&operation.mul_into(&eq_final[i2]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}

// given A,B,C,D: [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
// which is just batched logup: numerator and denominator ops
#[allow(non_snake_case)]
fn sum_eqspliteqlogup_hightolow<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E], trace3: &mut [E], trace4: &mut[E], alpha: E, z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {M/2-1+1}]:, [(); {M-M/2+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!([&trace1, &trace2, &trace3, &trace4].iter().all(|trace| trace.len() == 1<<M));
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_lowtohigh::<{M/2-1}, _>(z[M2..M-1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_lowtohigh::<{M-M/2}, _>(z[..M2].try_into().unwrap());
    let multieq1 = slicerev_multieqtable::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace1, &trace2, &trace3, &trace4);
        dbg!(&z);
        // dbg!(&trace1[..1<<M-1]);
        // dbg!(&trace1[1<<M-1..]);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part2,x_part1) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part2, x_part1)
    // with OP(x) = [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<M-round-1;
        // let z = z[M-1-round]; // be careful with override
        // dbg!(M-1-round);
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xM-i, zM-i)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie2 = i2;
                    let io2 = ie2 + step;
                    let ie = ie2 | (i1<<M2);
                    let io = io2 | (i1<<M2);
                    let (lowA, highA) = (trace1[ie], trace1[io]);
                    let (lowB, highB) = (trace2[ie], trace2[io]);
                    let (lowC, highC) = (trace3[ie], trace3[io]);
                    let (lowD, highD) = (trace4[ie], trace4[io]);
                    let subA = highA.sub_into(&lowA);
                    let subB = highB.sub_into(&lowB);
                    let subC = highC.sub_into(&lowC);
                    let subD = highD.sub_into(&lowD);
                    trace1[io] = subA;
                    trace2[io] = subB;
                    trace3[io] = subC;
                    trace4[io] = subD;
                    let lowRLC = lowA.add_into(&lowD.clone().mul_into(&alpha));
                    let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
                    let lin = lowB.mul_into(&subRLC).add_into(&lowRLC.mul_into(&subB)).add_into(&lowC.mul_into(&subD)).add_into(&lowD.mul_into(&subC));
                    let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!(":-)");
        // }
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M-1-round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M-1-round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M-1-round] = r;
        // fold A+B+C+D
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        if DEBUG {
            let layer_partA = &trace1[..1<<M-round-1];
            let layer_partB = &trace2[..1<<M-round-1];
            let layer_partC = &trace3[..1<<M-round-1];
            let layer_partD = &trace4[..1<<M-round-1];
            dbg!(layer_partA, layer_partB, layer_partC, layer_partD, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x_part2, x) EQ2(x_part2) EQ1(x) OP(x_part2, x)
    // with OP(x) = [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    debug_assert!(M-(M1-1) == M2+1);
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<M2-round;
        // let z = z[M2-round];
        // dbg!(M2-round);
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2;
                let io2 = ie2 + step;
                let ie = ie2;
                let io = io2;
                let (lowA,  highA) = (trace1[ie], trace1[io]);
                let (lowB,  highB) = (trace2[ie], trace2[io]);
                let (lowC,  highC) = (trace3[ie], trace3[io]);
                let (lowD,  highD) = (trace4[ie], trace4[io]);
                let subA = highA.sub_into(&lowA);
                let subB = highB.sub_into(&lowB);
                let subC = highC.sub_into(&lowC);
                let subD = highD.sub_into(&lowD);
                trace1[io] = subA;
                trace2[io] = subB;
                trace3[io] = subC;
                trace4[io] = subD;
                let lowRLC = lowA.add_into(&lowD.clone().mul_into(&alpha));
                let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
                let lin = lowB.mul_into(&subRLC).add_into(&lowRLC.mul_into(&subB)).add_into(&lowC.mul_into(&subD)).add_into(&lowD.mul_into(&subC));
                let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!("B-)");
        // }
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M2-round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M2-round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M2-round] = r;
        // fold A+B+C+D
        for i2 in 0..1<<M2-round {
            let ie2 = i2;
            let io2 = ie2 + step;
            let ie = ie2;
            let io = io2;
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let (lowC, subC) = (trace3[ie], trace3[io]);
            let (lowD, subD) = (trace4[ie], trace4[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            let foldC = lowC.add_into(&subC.mul_into(&r));
            let foldD = lowD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        if DEBUG {
            let layer_partA = &trace1[..1<<M2-round];
            let layer_partB = &trace2[..1<<M2-round];
            let layer_partC = &trace3[..1<<M2-round];
            let layer_partD = &trace4[..1<<M2-round];
            dbg!(layer_partA, layer_partB, layer_partC, layer_partD, claim_eqscale, claim_innereval);
            // dbg!(layer_partA.len());
        }
    }

    // let final_partA = &trace1[..1<<SKIPVARS];
    // let final_partB = &trace2[..1<<SKIPVARS];
    // send(&final_partA[..]);
    // send(&final_partB[..]);
    // // explicit sum
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(&final_partA, &final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[..SKIPVARS].try_into().unwrap());
        let mut sum = E::ZERO;
        for i in 0..1<<SKIPVARS {
            let operation = {
                let tmp = trace3[i].mul_into(&trace4[i]);
                trace2[i].mul_into(&trace1[i].add_into(&trace4[i].mul_into(&alpha))).add_into(&tmp)
            };
            // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
            sum = sum.add_into(&eq_final[i].mul_into(&operation));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}
// given A,B,C,D: [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
// which is just batched logup: numerator and denominator ops
#[allow(non_snake_case)]
fn sum_eqspliteqlogup_lowtohigh<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E], trace3: &mut [E], trace4: &mut[E], alpha: E, z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {(M/2-1)+1}]:, [(); {(M-M/2)+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!([&trace1, &trace2, &trace3, &trace4].iter().all(|trace| trace.len() == 1<<M));
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M/2-1}, _>(z[1..M1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M-M/2}, _>(z[M1..].try_into().unwrap());
    let multieq1 = slicerev_multieqtable_antiscattered::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable_antiscattered::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace1, &trace2, &trace3, &trace4);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part1,x_part2) EQ2(x_part2) * EQ1(x_part1) * OP(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * OP(x_part1, x_part2)
    // with OP(x) = [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<round;
        // let z = z[round]; // be careful with override
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO,E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lin_inner, mut sum_quad_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie1 = i1*step*2;
                    let io1 = ie1 + step;
                    let ie = ie1 | (i2<<M1);
                    let io = io1 | (i2<<M1);
                    let (evenA, oddA) = (trace1[ie], trace1[io]);
                    let (evenB, oddB) = (trace2[ie], trace2[io]);
                    let (evenC, oddC) = (trace3[ie], trace3[io]);
                    let (evenD, oddD) = (trace4[ie], trace4[io]);
                    let subA = oddA.sub_into(&evenA);
                    let subB = oddB.sub_into(&evenB);
                    let subC = oddC.sub_into(&evenC);
                    let subD = oddD.sub_into(&evenD);
                    trace1[io] = subA;
                    trace2[io] = subB;
                    trace3[io] = subC;
                    trace4[io] = subD;
                    let evenRLC = evenA.add_into(&evenD.clone().mul_into(&alpha));
                    let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
                    let lin = evenB.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subB)).add_into(&evenC.mul_into(&subD)).add_into(&evenD.mul_into(&subC));
                    let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
                    sum_lin_inner = sum_lin_inner.add_into(&lin.mul_into(&inner));
                    sum_quad_inner = sum_quad_inner.add_into(&quad.mul_into(&inner));
                }
                sum_lin = sum_lin.add_into(&sum_lin_inner.mul_into(&outer));
                sum_quad = sum_quad.add_into(&sum_quad_inner.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[round] = r; // update z in place
        // fold A+B+C+D
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            let (evenA, subA) = (trace1[ie], trace1[io]);
            let (evenB, subB) = (trace2[ie], trace2[io]);
            let (evenC, subC) = (trace3[ie], trace3[io]);
            let (evenD, subD) = (trace4[ie], trace4[io]);
            let foldA = evenA.add_into(&subA.mul_into(&r));
            let foldB = evenB.add_into(&subB.mul_into(&r));
            let foldC = evenC.add_into(&subC.mul_into(&r));
            let foldD = evenD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        if DEBUG {
            let layer_partA = trace1.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            let layer_partB = trace2.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            let layer_partC = trace3.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            let layer_partD = trace4.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            dbg!(layer_partA, layer_partB, layer_partC, layer_partD, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x, x_part2) EQ2(x_part2) EQ1(x) OP(x, x_part2)
    // with OP(x) = [A(x)*B(x) + C(x)*D(x)] + alpha*B(x)*D(x)
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<round;
        // let z = z[M1-1+round]; // be careful with override
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lin, mut sum_quad) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2*step*2;
                let io2 = ie2 + step;
                let ie = ie2<<M1-1;
                let io = io2<<M1-1;
                let (evenA, oddA) = (trace1[ie], trace1[io]);
                let (evenB, oddB) = (trace2[ie], trace2[io]);
                let (evenC, oddC) = (trace3[ie], trace3[io]);
                let (evenD, oddD) = (trace4[ie], trace4[io]);
                let subA = oddA.sub_into(&evenA);
                let subB = oddB.sub_into(&evenB);
                let subC = oddC.sub_into(&evenC);
                let subD = oddD.sub_into(&evenD);
                trace1[io] = subA;
                trace2[io] = subB;
                trace3[io] = subC;
                trace4[io] = subD;
                let evenRLC = evenA.add_into(&evenD.clone().mul_into(&alpha));
                let subRLC = subA.add_into(&subD.clone().mul_into(&alpha));
                let lin = evenB.mul_into(&subRLC).add_into(&evenRLC.mul_into(&subB)).add_into(&evenC.mul_into(&subD)).add_into(&evenD.mul_into(&subC));
                let quad = subB.mul_into(&subRLC).add_into(&subC.mul_into(&subD));
                sum_lin = sum_lin.add_into(&lin.mul_into(&outer));
                sum_quad = sum_quad.add_into(&quad.mul_into(&outer));
            }
            [sum_lin, sum_quad]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M1-1+round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M1-1+round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M1-1+round] = r; // update z with folding randomness
        // fold A+B+C+D
        for i2 in 0..1<<M2-round {
            let ie2 = i2*step*2;
            let io2 = ie2 + step;
            let ie = ie2<<M1-1;
            let io = io2<<M1-1;
            let (evenA, subA) = (trace1[ie], trace1[io]);
            let (evenB, subB) = (trace2[ie], trace2[io]);
            let (evenC, subC) = (trace3[ie], trace3[io]);
            let (evenD, subD) = (trace4[ie], trace4[io]);
            let foldA = evenA.add_into(&subA.mul_into(&r));
            let foldB = evenB.add_into(&subB.mul_into(&r));
            let foldC = evenC.add_into(&subC.mul_into(&r));
            let foldD = evenD.add_into(&subD.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
            trace3[ie] = foldC;
            trace4[ie] = foldD;
        }
        if DEBUG {
            let layer_partA = trace1.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            let layer_partB = trace2.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            let layer_partC = trace3.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            let layer_partD = trace4.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            dbg!(layer_partA, layer_partB, layer_partC, layer_partD, claim_eqscale, claim_innereval);
        }
    }

    // // explicit sum
    // let final_partA = trace1.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // let final_partB = trace2.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // send(final_partA.as_slice());
    // send(final_partB.as_slice());
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(final_partA, final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        // TODO: is it not better to just repeatedly fold?
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[M-SKIPVARS..].try_into().unwrap());
        let step = 1<<M-SKIPVARS;
        let mut sum = E::ZERO;
        for i2 in 0..1<<SKIPVARS {
            let i = i2*step;
            let operation = {
                let tmp = trace3[i].mul_into(&trace4[i]);
                trace2[i].mul_into(&trace1[i].add_into(&trace4[i].mul_into(&alpha))).add_into(&tmp)
            };
            // let operation = (trace1[i]*trace2[i]+trace3[i]*trace4[i]+alpha*trace2[i]*trace4[i]) % P;
            sum = sum.add_into(&eq_final[i2].mul_into(&operation));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}

#[allow(non_snake_case)]
fn sum_eqspliteqAB_hightolow<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E], z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {M/2-1+1}]:, [(); {M-M/2+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]: Sized,
{
    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_lowtohigh::<{M/2-1}, _>(z[M2..M-1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_lowtohigh::<{M-M/2}, _>(z[..M2].try_into().unwrap());
    let multieq1 = slicerev_multieqtable::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace1);
        dbg!(&trace2);
        dbg!(&z);
        // dbg!(&trace1[..1<<M-1]);
        // dbg!(&trace1[1<<M-1..]);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part2,x_part1) EQ2(x_part2) * EQ1(x_part1) * A(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * A(x_part2, x_part1)
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<M-round-1;
        // let z = z[M-1-round]; // be careful with override
        // dbg!(M-1-round);
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xM-i, zM-i)
            let (mut sum_lowsubs, mut sum_subs) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_lowsubs_inner, mut sum_subs_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie2 = i2;
                    let io2 = ie2 + step;
                    let ie = ie2 | (i1<<M2);
                    let io = io2 | (i1<<M2);
                    let (lowA, highA) = (trace1[ie], trace1[io]);
                    let (lowB, highB) = (trace2[ie], trace2[io]);
                    let subA = highA.sub_into(&lowA);
                    trace1[io] = subA;
                    let subB = highB.sub_into(&lowB);
                    trace2[io] = subB;
                    let lowsub = lowA.mul_into(&subB).add_into(&lowB.mul_into(&subA));
                    let sub = subA.mul_into(&subB);
                    sum_lowsubs_inner = sum_lowsubs_inner.add_into(&lowsub.mul_into(&inner));
                    sum_subs_inner = sum_subs_inner.add_into(&sub.mul_into(&inner));
                }
                sum_lowsubs = sum_lowsubs.add_into(&sum_lowsubs_inner.mul_into(&outer));
                sum_subs = sum_subs.add_into(&sum_subs_inner.mul_into(&outer));
            }
            [sum_lowsubs, sum_subs]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!(":-)");
        // }
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M-1-round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M-1-round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M-1-round] = r;
        // fold A+B
        for i in 0..1<<M-round-1 {
            let ie = i;
            let io = ie + step;
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        if DEBUG {
            let layer_partA = &trace1[..1<<M-round-1];
            let layer_partB = &trace2[..1<<M-round-1];
            dbg!(layer_partA, layer_partB, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x_part2, x) EQ2(x_part2) EQ1(x) A(x_part2, x)
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    debug_assert!(M-(M1-1) == M2+1);
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<M2-round;
        // let z = z[M2-round];
        // dbg!(M2-round);
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_lowsubs, mut sum_subs) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2;
                let io2 = ie2 + step;
                // let ie = ie2<<M1-1;
                // let io = io2<<M1-1;
                let ie = ie2;
                let io = io2;
                let (lowA,  highA) = (trace1[ie], trace1[io]);
                let (lowB,  highB) = (trace2[ie], trace2[io]);
                let subA = highA.sub_into(&lowA);
                trace1[io] = subA;
                let subB = highB.sub_into(&lowB);
                trace2[io] = subB;
                let lowsub = lowA.mul_into(&subB).add_into(&lowB.mul_into(&subA));
                let sub = subA.mul_into(&subB);
                sum_lowsubs = sum_lowsubs.add_into(&lowsub.mul_into(&outer));
                sum_subs = sum_subs.add_into(&sub.mul_into(&outer));
            }
            [sum_lowsubs, sum_subs]
        };
        tx.send(&g_inner);
        // now fold
        // if DEBUG {
        //     let one_minus_z = (P -z +1) % P;
        //     let g_0_inner = one_minus_z * sum_const % P;
        //     let g_1_inner = z*(sum_const+g_inner[0]+g_inner[1]) %P;
        //     dbg!(g_0_inner, g_1_inner, claim_innereval, claim_eqscale, claim);
        //     let g_0 = claim_eqscale * g_0_inner % P;
        //     let g_1 = claim_eqscale * g_1_inner % P;
        //     let oldclaim = claim_eqscale * claim_innereval % P;
        //     assert!((g_0 + g_1) % P == oldclaim);
        //     dbg!("B-)");
        // }
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M2-round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M2-round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M2-round] = r;
        // fold A+B
        for i2 in 0..1<<M2-round {
            let ie2 = i2;
            let io2 = ie2 + step;
            // let ie = ie2<<M1-1;
            // let io = io2<<M1-1;
            let ie = ie2;
            let io = io2;
            let (lowA, subA) = (trace1[ie], trace1[io]);
            let (lowB, subB) = (trace2[ie], trace2[io]);
            let foldA = lowA.add_into(&subA.mul_into(&r));
            let foldB = lowB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        if DEBUG {
            let layer_partA = &trace1[..1<<M2-round];
            let layer_partB = &trace2[..1<<M2-round];
            dbg!(layer_partA, layer_partB, claim_eqscale, claim_innereval);
            // dbg!(layer_partA.len());
        }
    }

    // let final_partA = &trace1[..1<<SKIPVARS];
    // let final_partB = &trace2[..1<<SKIPVARS];
    // send(&final_partA[..]);
    // send(&final_partB[..]);
    // // explicit sum
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(&final_partA, &final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[..SKIPVARS].try_into().unwrap());
        let mut sum = E::ZERO;
        for i in 0..1<<SKIPVARS {
            sum = sum.add_into(&eq_final[i].mul_into(&trace1[i]).mul_into(&trace2[i]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}
#[allow(non_snake_case)]
fn sum_eqspliteqAB_lowtohigh<const M: usize, E: FEXT, const SKIPVARS: usize>(tx: &mut Transcript<E>, trace1: &mut [E], trace2: &mut [E], z: &mut [E; M], claim: E)
where [(); M-2]:, // assert M >= 2 (easier to see compiler errors)
      [(); {M/2-1}]:, [(); {M-M/2}]:,
      [(); {(M/2-1)+1}]:, [(); {(M-M/2)+1}]:,
      [(); E::DEGREE]:, // TEMPORARY
      [(); 8/E::DEGREE]:,
{
    assert!(M >= 2, "size 2 tables not supported in eqsplit mode");
    assert!(trace1.len() == 1<<M && trace2.len() == 1<<M);
    let M1: usize = M/2;
    let M2: usize = M - M1;
    assert!(SKIPVARS <= M2+1, "ERROR: sumcheck SKIPVARS={SKIPVARS} but such a large skip doesn't make sense...");
    let multieq1_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M/2-1}, _>(z[1..M1].try_into().unwrap());
    let multieq2_raw = gen_multieqtable_recursive_hightolow_antiscatter::<{M-M/2}, _>(z[M1..].try_into().unwrap());
    let multieq1 = slicerev_multieqtable_antiscattered::<{M/2-1}, _>(&multieq1_raw);
    let multieq2 = slicerev_multieqtable_antiscattered::<{M-M/2}, _>(&multieq2_raw);
    debug_assert!(multieq1.len() == M1-1+1); // just safety
    debug_assert!(multieq2.len() == M2+1); // just safety
    if DEBUG {
        dbg!(&trace1);
        dbg!(&trace2);
    }
    let mut claim_innereval = claim;
    let mut claim_eqscale = E::ONE;

    // PHASE 1: sum_x=(x_part1,x_part2) EQ2(x_part2) * EQ1(x_part1) * A(x)
    //        = sum_(x_part2) EQ2(x_part2) sum_(x_part1) EQ1(x_part1) * A(x_part1, x_part2)
    if DEBUG {
        let info = format!("COMMENCING FIRST PHASE ({} ROUNDS)", M1-1); dbg!(info);
    }
    for round in 0..M1-1 {
        let step = 1<<round;
        // let z = z[round]; // be careful with override
        let eq2 = multieq2[0];
        let eq1_onelessvar = multieq1[round];
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", round+1, M-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_evensubs, mut sum_subs) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2 {
                let outer = eq2[i2]; // no need for folding or distinction between folding variable
                let (mut sum_evensubs_inner, mut sum_subs_inner) = (E::ZERO, E::ZERO);
                for i1 in 0..1<<M1-round-1 {
                    let inner = eq1_onelessvar[i1];
                    let ie1 = i1*step*2;
                    let io1 = ie1 + step;
                    let ie = ie1 | (i2<<M1);
                    let io = io1 | (i2<<M1);
                    let (evenA, oddA) = (trace1[ie], trace1[io]);
                    let (evenB, oddB) = (trace2[ie], trace2[io]);
                    let subA = oddA.sub_into(&evenA);
                    trace1[io] = subA;
                    let subB = oddB.sub_into(&evenB);
                    trace2[io] = subB;
                    let evensub = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
                    let sub = subA.mul_into(&subB);
                    sum_evensubs_inner = sum_evensubs_inner.add_into(&evensub.mul_into(&inner));
                    sum_subs_inner = sum_subs_inner.add_into(&sub.mul_into(&inner));
                }
                sum_evensubs = sum_evensubs.add_into(&sum_evensubs_inner.mul_into(&outer));
                sum_subs = sum_subs.add_into(&sum_subs_inner.mul_into(&outer));
            }
            [sum_evensubs, sum_subs]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[round];
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(&coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[round];
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[round] = r; // update z in place
        // fold A+B
        for i in 0..1<<M-round-1 {
            let ie = i*step*2;
            let io = ie + step;
            let (evenA, subA) = (trace1[ie], trace1[io]);
            let (evenB, subB) = (trace2[ie], trace2[io]);
            let foldA = evenA.add_into(&subA.mul_into(&r));
            let foldB = evenB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        if DEBUG {
            let layer_partA = trace1.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            let layer_partB = trace2.iter_mut().step_by(step<<1).collect::<Vec<&mut E>>();
            dbg!(layer_partA, layer_partB, claim_eqscale, claim_innereval);
        }
    }

    // PHASE 2: sum_(x, x_part2) EQ2(x_part2) EQ1(x) A(x, x_part2)
    if DEBUG {
        let info = format!("COMMENCING SECOND PHASE ({} ROUNDS)", 1+M2-SKIPVARS); dbg!(info);
    }
    for round in 0..1+M2-SKIPVARS {
        let step = 1<<round;
        // let z = z[M1-1+round]; // be careful with override
        let eq2_onelessvar = multieq2[round]; // last round will be [1]
        debug_assert!(eq2_onelessvar.len() == 1<<M2-round);
        if DEBUG {
            let info = format!("sumcheck round {}: folding layers of log(size) {}", M1+round, 1+M2-round); dbg!(info);
        }
        let g_inner = {
            // compute non-constant coefficients modulo eq(xi, zi)
            let (mut sum_evensubs, mut sum_subs) = (E::ZERO, E::ZERO);
            for i2 in 0..1<<M2-round {
                let outer = eq2_onelessvar[i2];
                let ie2 = i2*step*2;
                let io2 = ie2 + step;
                let ie = ie2<<M1-1;
                let io = io2<<M1-1;
                let (evenA, oddA) = (trace1[ie], trace1[io]);
                let (evenB, oddB) = (trace2[ie], trace2[io]);
                let subA = oddA.sub_into(&evenA);
                trace1[io] = subA;
                let subB = oddB.sub_into(&evenB);
                trace2[io] = subB;
                let evensub = evenA.mul_into(&subB).add_into(&evenB.mul_into(&subA));
                let sub = subA.mul_into(&subB);
                sum_evensubs = sum_evensubs.add_into(&evensub.mul_into(&outer));
                sum_subs = sum_subs.add_into(&sub.mul_into(&outer));
            }
            [sum_evensubs, sum_subs]
        };
        tx.send(&g_inner);
        // now fold
        let r: E = tx.rand();
        claim_innereval = {
            let z = z[M1-1+round]; // be careful with override
            if g_inner.len() == 0 {
                unreachable!()
            } else if g_inner.len() == 1 {
                let linear = g_inner[0];
                let diff = r.clone().sub_into(&z);
                claim_innereval.add_into(&diff.mul_into(&linear))
            } else {
                let g_inner_1 = g_inner.iter().fold(E::ZERO, |acc, x| acc.add_into(x));
                let g_inner_r = { // horner
                    let mut out = E::ZERO;
                    for coeff in g_inner.iter().rev() {
                        out = out.add_into(coeff).mul_into(&r);
                    }
                    out
                };
                claim_innereval.add_into(&g_inner_r.sub_into(&g_inner_1.mul_into(&z)))
            }
        };
        claim_eqscale = {
            let z = z[M1-1+round]; // be careful with override
            let one_minus_z = E::ONE.sub_into(&z);
            let z_minus_one_minus_z = z.sub_into(&one_minus_z);
            let eq_r = one_minus_z.add_into(&z_minus_one_minus_z.mul_into(&r));
            claim_eqscale.mul_into(&eq_r)
        };
        z[M1-1+round] = r; // update z with folding randomness
        // fold A+B
        for i2 in 0..1<<M2-round {
            let ie2 = i2*step*2;
            let io2 = ie2 + step;
            let ie = ie2<<M1-1;
            let io = io2<<M1-1;
            let (evenA, subA) = (trace1[ie], trace1[io]);
            let (evenB, subB) = (trace2[ie], trace2[io]);
            let foldA = evenA.add_into(&subA.mul_into(&r));
            let foldB = evenB.add_into(&subB.mul_into(&r));
            trace1[ie] = foldA;
            trace2[ie] = foldB;
        }
        if DEBUG {
            let layer_partA = trace1.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            let layer_partB = trace2.iter_mut().step_by(step<<M1).collect::<Vec<&mut E>>();
            dbg!(layer_partA, layer_partB, claim_eqscale, claim_innereval);
        }
    }

    // // explicit sum
    // let final_partA = trace1.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // let final_partB = trace2.iter().step_by(1<<M-SKIPVARS).copied().collect::<Vec<u8>>();
    // send(final_partA.as_slice());
    // send(final_partB.as_slice());
    if DEBUG {
        let info = "VERIFIER PHASE: testing full sum evaluation..."; dbg!(info);
        // dbg!(final_partA, final_partB);
    }
    if true || claim_eqscale != E::ZERO { // we can skip sum when scale goes to 0
        // TODO: is it not better to just repeatedly fold?
        let eq_final = gen_eqtable_recursive_lowtohigh::<{SKIPVARS}, _>(z[M-SKIPVARS..].try_into().unwrap());
        let step = 1<<M-SKIPVARS;
        let mut sum = E::ZERO;
        for i2 in 0..1<<SKIPVARS {
            let i = i2*step;
            sum = sum.add_into(&eq_final[i2].mul_into(&trace1[i]).mul_into(&trace2[i]));
        }
        assert!(sum == claim_innereval);
        if DEBUG {
            let info = "good sumcheck!"; dbg!(info);
        }
    } else {
        if DEBUG {
            let info = "EQ SCALING FACTOR == 0, SKIPPING CHECKS"; dbg!(info);
        }
    }
}