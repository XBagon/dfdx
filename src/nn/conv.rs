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
        const IC: usize,
        const IH: usize,
        const IW: usize,
        const OC: usize,
        const K: usize,
        const S: usize,
        const P: usize,
        H: Tape,
    > Module<Tensor3D<IC, IH, IW, H>> for Conv2D<IC, OC, K, S, P>
where
    [(); out_size(IW, P, K, S)]: Sized,
    [(); out_size(IH, P, K, S)]: Sized,
    [(); IC * K * K]: Sized,
    [(); out_size(IH, P, K, S) * out_size(IW, P, K, S)]: Sized,
{
    type Output = Tensor3D<OC, { out_size(IH, P, K, S) }, { out_size(IW, P, K, S) }, H>;

    fn forward(&self, input: Tensor3D<IC, IH, IW, H>) -> Self::Output {
        let (input, tape) = input.split_tape();
        // let mut col = Cpu::zeros();
        // im2col::<IC, IH, IW, { out_size(IH, P, K, S) }, { out_size(IW, P, K, S) }, K, S, P>(
        //     input.data(),
        //     col.as_mut(),
        // );
        todo!();
    }
}

fn im2col<
    const IN_CHANNELS: usize,
    const IN_HEIGHT: usize,
    const IN_WIDTH: usize,
    const OUT_HEIGHT: usize,
    const OUT_WIDTH: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
>(
    im: &[[[f32; IN_WIDTH]; IN_HEIGHT]; IN_CHANNELS],
    col: &mut [[f32; OUT_HEIGHT * OUT_WIDTH]; IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE],
) where
    [(); IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]: Sized,
    [(); OUT_HEIGHT * OUT_WIDTH]: Sized,
{
    for col_0 in 0..IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE {
        let im_channel = col_0 / (KERNEL_SIZE * KERNEL_SIZE);
        let in_kernel_ind = col_0 % (KERNEL_SIZE * KERNEL_SIZE);
        let h_offset = in_kernel_ind / KERNEL_SIZE;
        let w_offset = in_kernel_ind % KERNEL_SIZE;
        for col_1 in 0..OUT_HEIGHT * OUT_WIDTH {
            debug_assert_eq!(col[col_0][col_1], 0.0);
            let im_height = ((col_1 / OUT_WIDTH) * STRIDE + h_offset).checked_sub(PADDING);
            let im_width = ((col_1 % OUT_WIDTH) * STRIDE + w_offset).checked_sub(PADDING);
            col[col_0][col_1] = match (im_height, im_width) {
                (Some(h), Some(w)) if h < IN_HEIGHT && w < IN_WIDTH => im[im_channel][h][w],
                _ => 0.0,
            };
        }
    }
}

fn col2im<
    const IN_CHANNELS: usize,
    const IN_HEIGHT: usize,
    const IN_WIDTH: usize,
    const OUT_HEIGHT: usize,
    const OUT_WIDTH: usize,
    const KERNEL_SIZE: usize,
    const STRIDE: usize,
    const PADDING: usize,
>(
    col: &[[f32; OUT_HEIGHT * OUT_WIDTH]; IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE],
    im: &mut [[[f32; IN_WIDTH]; IN_HEIGHT]; IN_CHANNELS],
) where
    [(); IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE]: Sized,
    [(); OUT_HEIGHT * OUT_WIDTH]: Sized,
{
    for col_0 in 0..IN_CHANNELS * KERNEL_SIZE * KERNEL_SIZE {
        let im_channel = col_0 / (KERNEL_SIZE * KERNEL_SIZE);
        let in_kernel_ind = col_0 % (KERNEL_SIZE * KERNEL_SIZE);
        let h_offset = in_kernel_ind / KERNEL_SIZE;
        let w_offset = in_kernel_ind % KERNEL_SIZE;
        for col_1 in 0..OUT_HEIGHT * OUT_WIDTH {
            let im_height = ((col_1 / OUT_WIDTH) * STRIDE + h_offset).checked_sub(PADDING);
            let im_width = ((col_1 % OUT_WIDTH) * STRIDE + w_offset).checked_sub(PADDING);
            match (im_height, im_width) {
                (Some(h), Some(w)) if h < IN_HEIGHT && w < IN_WIDTH => {
                    im[im_channel][h][w] = col[col_0][col_1]
                }
                _ => {
                    debug_assert_eq!(col[col_0][col_1], 0.0);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_im2col() {
        // test using torch.nn.Unfold(2)
        let image = [
            [[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]],
            [[-1.0, -2.0, -3.0, -4.0], [-6.0, -7.0, -8.0, -9.0]],
            [[0.0, 0.1, 0.2, 0.3], [0.5, 0.6, 0.7, 0.8]],
        ];
        let mut out = [[0.0f32; 3]; 12];

        im2col::<3, 2, 4, 1, 3, 2, 1, 0>(&image, &mut out);
        assert_eq!(
            &out,
            &[
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [6.0, 7.0, 8.0],
                [7.0, 8.0, 9.0],
                [-1.0, -2.0, -3.0],
                [-2.0, -3.0, -4.0],
                [-6.0, -7.0, -8.0],
                [-7.0, -8.0, -9.0],
                [0.0, 0.1, 0.2],
                [0.1, 0.2, 0.3],
                [0.5, 0.6, 0.7],
                [0.6, 0.7, 0.8]
            ]
        );
    }

    #[test]
    fn test_col2im_reverses_im2col() {
        let image = [
            [[1.0, 2.0, 3.0, 4.0], [6.0, 7.0, 8.0, 9.0]],
            [[-1.0, -2.0, -3.0, -4.0], [-6.0, -7.0, -8.0, -9.0]],
            [[0.0, 0.1, 0.2, 0.3], [0.5, 0.6, 0.7, 0.8]],
        ];

        let mut col = [[0.0f32; 3]; 12];
        im2col::<3, 2, 4, 1, 3, 2, 1, 0>(&image, &mut col);

        let mut image2 = [[[0.0f32; 4]; 2]; 3];
        col2im::<3, 2, 4, 1, 3, 2, 1, 0>(&col, &mut image2);

        assert_eq!(image, image2);
    }
}
