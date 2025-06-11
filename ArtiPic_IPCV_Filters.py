import numpy as np
from PIL import Image
import streamlit as st

# ASCII characters for 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1[]?-_+~i!lI;:,^`'. "
# ASCII characters for 10 levels of gray
gscale2 = '@%#*+=-:. '

# Calculate average luminance of an image region
def getAverageL(image):
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w * h))

# Mean filter
def mean_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.mean(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

# Min filter
def min_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.min(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

# Max filter
def max_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.max(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

# Sobel filter
def sobel_filter(image):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    Ix = np.zeros_like(image, dtype=float)
    Iy = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+3, j:j+3]
            Ix[i, j] = np.sum(Kx * region)
            Iy[i, j] = np.sum(Ky * region)
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    return G.astype(np.uint8)

# Gaussian filter
def gaussian_filter(image, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered_img[i, j] = np.sum(region * kernel)
    filtered_img = np.clip(filtered_img, 0, 255)
    return filtered_img.astype(np.uint8)

# Convert image to ASCII art
def covertImageToAscii(image, cols, scale, moreLevels):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    W, H = image.size[0], image.size[1]
    w = W / cols
    h = w / scale
    rows = int(H / h)
    aimg = []
    for j in range(rows):
        y1 = int(j * h)
        y2 = int((j + 1) * h)
        if j == rows - 1:
            y2 = H
        aimg.append("")
        for i in range(cols):
            x1 = int(i * w)
            x2 = int((i + 1) * w)
            if i == cols - 1:
                x2 = W
            img = image.crop((x1, y1, x2, y2))
            avg = int(getAverageL(img))
            if moreLevels:
                gsval = gscale1[int((avg * (len(gscale1) - 1)) / 255)]
            else:
                gsval = gscale2[int((avg * (len(gscale2) - 1)) / 255)]
            aimg[j] += gsval
    return aimg

# Main Streamlit app
def main():
    st.title("Image to ASCII Art with Custom Filters")

    img_file = st.file_uploader("Choose a photo:", type="jpg")

    if img_file is not None:
        image = Image.open(img_file).convert('L')
        st.subheader('Original Image')
        st.image(image, caption='Original', use_column_width=True)

        filter_options = {
            'mean': mean_filter,
            'min': min_filter,
            'max': max_filter,
            'sobel': sobel_filter,
            'gaussian': gaussian_filter
        }

        selected_filters = st.multiselect('Select filters to apply in order', list(filter_options.keys()))

        if selected_filters:
            processed_imgs = []
            current_img = np.array(image)
            for filter_name in selected_filters:
                filter_func = filter_options[filter_name]
                current_img = filter_func(current_img)
                processed_imgs.append(current_img)

            for i, filter_name in enumerate(selected_filters):
                st.subheader(f'After {filter_name} Filter')
                st.image(processed_imgs[i], caption=f'After {filter_name}', use_column_width=True)

            if st.button('Convert to ASCII Art'):
                final_img = processed_imgs[-1] if processed_imgs else np.array(image)
                aimg = covertImageToAscii(final_img, cols=100, scale=0.43, moreLevels=False)
                st.text('\n'.join(aimg))
        else:
            st.info("Select at least one filter to process the image.")

if __name__ == '__main__':
    main()