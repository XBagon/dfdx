pub use crate::prelude::*;

// (x + 2 * PADDING - KERNEL_SIZE) / STRIDE + 1
pub const fn out_size(in_size: usize, padding: usize, kernel_size: usize, stride: usize) -> usize {
    (in_size + 2 * padding - kernel_size) / stride + 1
}

/// Conv2D<1, 3, 3>
#[derive(Clone, Debug, Default)]
pub struct Conv2D<
    const IN_CHANNELS: usize,
    const OUT_CHANNELS: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize = 1,
    const PADDING: usize = 0,
> {
    weight: Tensor4D<OUT_CHANNELS, IN_CHANNELS, KERNEL_SIZE, KERNEL_SIZE>,
    bias: Tensor1D<OUT_CHANNELS>,
}

impl<
        const IN_CHANNELS: usize,
        const OUT_CHANNELS: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize,
        const PADDING: usize,
    > CanUpdateWithGradients for Conv2D<IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.weight.update(grads);
        self.bias.update(grads);
    }
}

impl<
        const IN_CHANNELS: usize,
        const IN_HEIGHT: usize,
        const IN_WIDTH: usize,
        const OUT_CHANNELS: usize,
        const KERNEL_SIZE: usize,
        const STRIDE: usize,
        const PADDING: usize,
        H: Tape,
    > Module<Tensor3D<IN_CHANNELS, IN_HEIGHT, IN_WIDTH, H>>
    for Conv2D<IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING>
where
    [(); out_size(IN_WIDTH, PADDING, KERNEL_SIZE, STRIDE)]: Sized,
    [(); out_size(IN_HEIGHT, PADDING, KERNEL_SIZE, STRIDE)]: Sized,
    [(); IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]: Sized,
    [(); {
        out_size(IN_HEIGHT, PADDING, KERNEL_SIZE, STRIDE)
            * out_size(IN_WIDTH, PADDING, KERNEL_SIZE, STRIDE)
    }]: Sized,
{
    type Output = Tensor3D<
        OUT_CHANNELS,
        { out_size(IN_HEIGHT, PADDING, KERNEL_SIZE, STRIDE) },
        { out_size(IN_WIDTH, PADDING, KERNEL_SIZE, STRIDE) },
        H,
    >;
    fn forward(&self, input: Tensor3D<IN_CHANNELS, IN_HEIGHT, IN_WIDTH, H>) -> Self::Output {
        let (input, tape) = input.split_tape();
        let col = im2col::<
            IN_CHANNELS,
            IN_HEIGHT,
            IN_WIDTH,
            OUT_CHANNELS,
            { out_size(IN_HEIGHT, PADDING, KERNEL_SIZE, STRIDE) },
            { out_size(IN_WIDTH, PADDING, KERNEL_SIZE, STRIDE) },
            KERNEL_SIZE,
            STRIDE,
            PADDING,
        >(input);
        todo!();
    }
}

pub fn im2col<
    const IN_CHANNELS: usize,
    const IN_HEIGHT: usize,
    const IN_WIDTH: usize,
    const OUT_CHANNELS: usize,
    const OUT_HEIGHT: usize,
    const OUT_WIDTH: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
>(
    im: Tensor3D<IN_CHANNELS, IN_HEIGHT, IN_WIDTH>,
) -> Tensor2D<{ IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE }, { OUT_HEIGHT * OUT_WIDTH }>
where
    [(); IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]: Sized,
    [(); OUT_HEIGHT * OUT_WIDTH]: Sized,
{
    todo!();
}
