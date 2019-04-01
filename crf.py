import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
import numpy as np
import matplotlib.pyplot as plt

H, W, NLABELS = 512, 512, 2

img = np.load("4_true_img.np.npy")
soft_max = np.load("4_pred_mask.np.npy")
#soft_max = np.zeros((H,W))



soft_max = (soft_max-soft_max.min()) / (soft_max.max()-soft_max.min())
soft_max = 0.5 + 0.2 * (soft_max-0.5)

soft_max = np.tile(soft_max[np.newaxis,:,:],(2,1,1))
soft_max[1,:,:] = 1 - soft_max[0,:,:]

U = unary_from_softmax(soft_max)  # note: num classes is first dim

# d = dcrf.DenseCRF2D(W, H, NLABELS)
# d.setUnaryEnergy(U)

# Run inference for 10 iterations
# Q_unary = d.inference(10)

# # The Q is now the approximate posterior, we can get a MAP estimate using argmax.
# map_soln_unary = np.argmax(Q_unary, axis=0)

# # Unfortunately, the DenseCRF flattens everything, so get it back into picture form.
# map_soln_unary = map_soln_unary.reshape((H,W))

# # And let's have a look.
# plt.imshow(map_soln_unary); plt.axis('off'); plt.title('MAP Solution without pairwise terms');


# Create simple image which will serve as bilateral.
# Note that we put the channel dimension last here,
# but we could also have it be the first dimension and
# just change the `chdim` parameter to `0` further down.
img = img[:,:,np.newaxis]


#print(img.shape)

pairwise_energy = create_pairwise_bilateral(sdims=(10,10), schan=(0.01,), img=img, chdim=2)
###################
###################
d = dcrf.DenseCRF2D(W, H, NLABELS)
d.setUnaryEnergy(U)
d.addPairwiseEnergy(pairwise_energy, compat=10)  # `compat` is the "strength" of this potential.

# This time, let's do inference in steps ourselves
# so that we can look at intermediate solutions
# as well as monitor KL-divergence, which indicates
# how well we have converged.
# PyDenseCRF also requires us to keep track of two
# temporary buffers it needs for computations.
Q, tmp1, tmp2 = d.startInference()
for _ in range(5):
    d.stepInference(Q, tmp1, tmp2)
kl1 = d.klDivergence(Q) / (H*W)
map_soln1 = np.argmax(Q, axis=0).reshape((H,W))

for _ in range(20):
    d.stepInference(Q, tmp1, tmp2)
kl2 = d.klDivergence(Q) / (H*W)
map_soln2 = np.argmax(Q, axis=0).reshape((H,W))

for _ in range(50):
    d.stepInference(Q, tmp1, tmp2)
kl3 = d.klDivergence(Q) / (H*W)
map_soln3 = np.argmax(Q, axis=0).reshape((H,W))

img_en = pairwise_energy.reshape((-1, H, W))  # Reshape just for plotting
plt.figure(figsize=(15,5))
plt.subplot(1,3,1); plt.imshow(map_soln1);
plt.title('MAP Solution with DenseCRF\n(5 steps, KL={:.2f})'.format(kl1)); plt.axis('off');
plt.subplot(1,3,2); plt.imshow(map_soln2);
plt.title('MAP Solution with DenseCRF\n(20 steps, KL={:.2f})'.format(kl2)); plt.axis('off');
plt.subplot(1,3,3); plt.imshow(map_soln3);
plt.title('MAP Solution with DenseCRF\n(75 steps, KL={:.2f})'.format(kl3)); plt.axis('off');

plt.savefig('check.png')