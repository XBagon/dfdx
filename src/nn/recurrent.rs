use std::borrow::Borrow;
use std::cell::Cell;
use crate::prelude::*;

pub struct Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>,
        F: Module<(TIn, TOut), Output = TOut>,
{
    inner: F,
    memory: Cell<F::Output>
}

impl<TIn, TOut, F: Default> Default for Recurrent<TIn, TOut, F> where
    TOut: Tensor<Dtype = f32>,
    TIn: (Tensor<Dtype = f32>),
    TOut: Tensor<Dtype = f32> + Default,
    F: Module<(TIn, TOut), Output = TOut>,
{
    fn default() -> Self {
        Self {
            inner: Default::default(),
            memory: Cell::new(Default::default()),
        }
    }
}

impl<TIn, TOut, F: CanUpdateWithGradients> CanUpdateWithGradients for Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>,
        F: Module<(TIn, TOut), Output = TOut>,
{
    /// Pass through to `F`'s [CanUpdateWithGradients].
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        self.inner.update(grads);
    }
}

impl<TIn, TOut, F: ResetParams> ResetParams for Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>,
        F: Module<(TIn, TOut), Output = TOut>,
{
    /// Pass through to `F`'s [ResetParams].
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.inner.reset_params(rng);
    }
}

impl<TIn, TOut, F> Module<TIn> for Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>, TOut::Tape: Default,
        F: Module<(TIn, TOut), Output = TOut>,
{
    type Output = F::Output;

    /// Calls forward on `F` and then adds `x` to the result: `F(x) + x`
    fn forward(&self, x: TIn) -> Self::Output {
        let (x, tape) = x.split_tape();
        let recurrent_x = x;
        let y = self.inner.forward((x.duplicate().put_tape(tape), self.memory.borrow().clone().into_inner()));
        self.memory.set(y.duplicate().put_tape(TOut::Tape::default()));
        y
    }
}

impl<TIn, TOut, F: SaveToNpz> SaveToNpz for Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>,
        F: Module<(TIn, TOut), Output = TOut>,
{
    /// Pass through to `F`'s [SaveToNpz].
    fn write<W>(
        &self,
        filename_prefix: &str,
        w: &mut zip::ZipWriter<W>,
    ) -> zip::result::ZipResult<()>
        where
            W: std::io::Write + std::io::Seek,
    {
        self.inner.write(filename_prefix, w)?;
        Ok(())
    }
}

impl<TIn, TOut, F: LoadFromNpz> LoadFromNpz for Recurrent<TIn, TOut, F>
    where
        TIn: Tensor<Dtype = f32>,
        TOut: Tensor<Dtype = f32>,
        F: Module<(TIn, TOut), Output = TOut>,
{
    /// Pass through to `F`'s [LoadFromNpz].
    fn read<R>(&mut self, filename_prefix: &str, r: &mut zip::ZipArchive<R>) -> Result<(), NpzError>
        where
            R: std::io::Read + std::io::Seek,
    {
        self.inner.read(filename_prefix, r)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::assert_close;
    use rand::{prelude::StdRng, SeedableRng};
    use std::fs::File;
    use tempfile::NamedTempFile;
    use zip::ZipArchive;
    use crate::nn::concat::Concat;

    #[test]
    fn test_reset() {
        let mut rng = StdRng::seed_from_u64(0);
        let mut model: Recurrent<Tensor1D<2>, Tensor1D<5>, (Concat, Linear<2, 5>)> = Default::default();
        assert_eq!(model.inner.weight.data(), &[[0.0; 2]; 5]);
        assert_eq!(model.inner.bias.data(), &[0.0; 5]);

        model.reset_params(&mut rng);
        assert_ne!(model.inner.weight.data(), &[[0.0; 2]; 5]);
        assert_ne!(model.inner.bias.data(), &[0.0; 5]);
    }
}