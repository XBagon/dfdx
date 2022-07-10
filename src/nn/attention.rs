use crate::prelude::*;
use crate::tensor_ops::utils::move_tape_and_add_backward_op;

#[derive(Debug)]
pub struct MultiHeadAttention<
    const M: usize,
    const K: usize,
    const V: usize,
    const N: usize,
    const NK: usize,
    const NV: usize,
> {
    w_qkv: SplitInto<(Linear<M, NK>, Linear<M, NK>, Linear<M, NV>)>,
    w_out: Linear<NV, M>,
}

impl<
        const M: usize,
        const K: usize,
        const V: usize,
        const N: usize,
        const NK: usize,
        const NV: usize,
    > ResetParams for MultiHeadAttention<M, K, V, N, NK, NV>
{
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.w_qkv.reset_params(rng);
        self.w_out.reset_params(rng);
    }
}

impl<
        const M: usize,
        const K: usize,
        const V: usize,
        const N: usize,
        const NK: usize,
        const NV: usize,
    > CanUpdateWithGradients for MultiHeadAttention<M, K, V, N, NK, NV>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.w_qkv.update(grads);
        self.w_out.update(grads);
    }
}

impl<
        const M: usize,
        const K: usize,
        const V: usize,
        const N: usize,
        const NK: usize,
        const NV: usize,
        TAPE: Tape,
    > Module<Tensor1D<M, TAPE>> for MultiHeadAttention<M, K, V, N, NK, NV>
{
    type Output = Tensor1D<M, TAPE>;
    fn forward(&self, x: Tensor1D<M, TAPE>) -> Self::Output {
        let (q, k, v) = self.w_qkv.forward(x);

        let v: Tensor2D<N, V, TAPE> = reshape(v);
        let (v, tape) = v.split_tape();

        let k: Tensor2D<N, K, TAPE> = reshape(k.put_tape(tape));
        let (k, tape) = k.split_tape();

        let q: Tensor2D<N, K, TAPE> = reshape(q.put_tape(tape));

        let qk: Tensor2D<N, N, TAPE> = matmul_transpose(q, &k);
        let qk: Tensor2D<N, N, TAPE> = softmax(qk / (K as f32).sqrt());
        let qkv: Tensor2D<N, V, TAPE> = matmul(qk, &v);
        let r: Tensor1D<NV, TAPE> = reshape(qkv);
        self.w_out.forward(r)
    }
}

impl<
        const B: usize,
        const M: usize,
        const K: usize,
        const V: usize,
        const N: usize,
        const NK: usize,
        const NV: usize,
        TAPE: Tape,
    > Module<Tensor2D<B, M, TAPE>> for MultiHeadAttention<M, K, V, N, NK, NV>
{
    type Output = Tensor2D<B, M, TAPE>;
    fn forward(&self, x: Tensor2D<B, M, TAPE>) -> Self::Output {
        let (q, k, v) = self.w_qkv.forward(x);

        let v: Tensor3D<B, N, V, TAPE> = reshape(v);
        let (v, tape) = v.split_tape();

        let k: Tensor3D<B, N, K, TAPE> = reshape(k.put_tape(tape));
        let (k, tape) = k.split_tape();

        let q: Tensor3D<B, N, K, TAPE> = reshape(q.put_tape(tape));

        let qk: Tensor3D<B, N, N, TAPE> = batch_matmul_transpose(q, &k);
        let qk: Tensor3D<B, N, N, TAPE> = softmax(qk / (K as f32).sqrt());
        let qkv: Tensor3D<B, N, V, TAPE> = batch_matmul(qk, &v);
        let r: Tensor2D<B, NV, TAPE> = reshape(qkv);
        self.w_out.forward(r)
    }
}

fn reshape<T: Tensor<Dtype = f32>, R: Tensor<Dtype = f32, Tape = T::Tape>>(src: T) -> R {
    assert_eq!(T::Array::NUM_ELEMENTS, R::Array::NUM_ELEMENTS);
    let mut dst = R::NoTape::zeros();
    copy(src.data(), dst.mut_data());
    move_tape_and_add_backward_op(src, dst, move |mut src, dst, grads| {
        let (src_grad, dst_grad) = grads.mut_and_ref(&src, &dst);
        copy(dst_grad, src.mut_data());
        T::Device::add(src_grad, src.data());
    })
}

fn copy<Src: CountElements, Dst: CountElements<Dtype = Src::Dtype>>(src: &Src, dst: &mut Dst) {
    let num_elements = Src::NUM_ELEMENTS.min(Dst::NUM_ELEMENTS);
    let src_ptr = src.ref_first_elem() as *const Src::Dtype;
    let dst_ptr = dst.mut_first_elem() as *mut Dst::Dtype;
    unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, num_elements) }
}
