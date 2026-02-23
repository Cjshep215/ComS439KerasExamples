from setup import *
import nerf_core

images, poses, focal = nerf_core.load_data()

# # Plot a random image from the dataset for visualization.
# plt.imshow(images[np.random.randint(low=0, high=num_images)])
# plt.show()

train_ds, val_ds = nerf_core.build_datasets(images, poses, focal)
model = nerf_core.train_nerf(train_ds, val_ds)
nerf_core.render_results(model, images, poses, focal)