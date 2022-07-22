use rand::Rng;
use crate::prelude::*;

pub struct Concat {

}

impl CanUpdateWithGradients for Concat {
    fn update<G: GradientProvider>(&mut self, grads: &mut G) {
        todo!()
    }
}

impl ResetParams for Concat {
    fn reset_params<R: Rng>(&mut self, rng: &mut R) {
        todo!()
    }
}

impl<const A: usize, const B: usize> Module<(Tensor1D<A>, Tensor1D<B>)> for Concat where [(); {A+B}]:, {
    type Output = Tensor1D<{A+B}>;

    fn forward(&self, x: (Tensor1D<A>, Tensor1D<B>)) -> Self::Output {
        let mut y = [0.; {A+B}];
        for (i, x_i) in x.0.data().into_iter().chain(x.1.data().into_iter()).enumerate() {
            y[i] = *x_i;
        }
        Tensor1D::new(y)
    }
}