import numpy as np
from PIL import Image
import streamlit as st
from scipy import fftpack

# Gray scale level values
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1[]?-_+~i!lI;:,^`'. "
gscale2 = '@#*+=-:,.  '
custom_scales = {
    'Original 70-level': gscale1,
    'Original 10-level': gscale2,
    'Dark Scale 1': '@%#*+=-:,. ',
    'Dark Scale 2': '@#*+=-:,.  ',
    'Dark Scale 3': '@#$%*+=:,. ',
    'Dark Scale 4': '@#%*+=-:;  ',
    'Dark Scale 5': '@#*+=-:;,. '
}

def getAverageL(image):
    im = np.array(image)
    w, h = im.shape
    return np.average(im.reshape(w * h))

# Cached Filter Functions
@st.cache_data
def mean_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.mean(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

@st.cache_data
def min_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.min(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

@st.cache_data
def max_filter(image, kernel_size=3):
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_img[i, j] = np.max(padded_img[i:i+kernel_size, j:j+kernel_size])
    return filtered_img.astype(np.uint8)

@st.cache_data
def sobel_filter(image):
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = np.zeros_like(image, dtype=float)
    Iy = np.zeros_like(image, dtype=float)
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+3, j:j+3]
            Ix[i, j] = np.sum(Kx * region)
            Iy[i, j] = np.sum(Ky * region)
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    return G.astype(np.uint8)

@st.cache_data
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

@st.cache_data
def butterworth_filter(image, cutoff=30, order=2):
    f = fftpack.fft2(image)
    fshift = fftpack.fftshift(f)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows)
    v = np.arange(cols)
    V, U = np.meshgrid(v, u)
    D = np.sqrt((U - crow)**2 + (V - ccol)**2)
    H = 1 / (1 + (D / cutoff)**(2 * order))
    fshift_filtered = fshift * H
    f_ishift = fftpack.ifftshift(fshift_filtered)
    img_back = fftpack.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = (img_back / img_back.max()) * 255
    return img_back.astype(np.uint8)

@st.cache_data
def prewitt_filter(image):
    Kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    Ix = np.zeros_like(image, dtype=float)
    Iy = np.zeros_like(image, dtype=float)
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+3, j:j+3]
            Ix[i, j] = np.sum(Kx * region)
            Iy[i, j] = np.sum(Ky * region)
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    return G.astype(np.uint8)

@st.cache_data
def robert_filter(image):
    Kx = np.array([[1, 0], [0, -1]])
    Ky = np.array([[0, 1], [-1, 0]])
    Ix = np.zeros_like(image, dtype=float)
    Iy = np.zeros_like(image, dtype=float)
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+2, j:j+2]
            Ix[i, j] = np.sum(Kx * region)
            Iy[i, j] = np.sum(Ky * region)
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    return G.astype(np.uint8)

@st.cache_data
def laplacian_filter(image):
    K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    filtered_img = np.zeros_like(image, dtype=float)
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+3, j:j+3]
            filtered_img[i, j] = np.sum(K * region)
    filtered_img = np.abs(filtered_img)
    filtered_img = (filtered_img / filtered_img.max()) * 255
    return filtered_img.astype(np.uint8)

@st.cache_data
def mexican_hat_filter(image, kernel_size=5, sigma=1.0):
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = (xx**2 + yy**2 - 2*sigma**2) / (sigma**4) * np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))  # Normalize
    pad = kernel_size // 2
    padded_img = np.pad(image, pad, mode='edge')
    filtered_img = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            filtered_img[i, j] = np.sum(region * kernel)
    filtered_img = np.abs(filtered_img)
    filtered_img = (filtered_img / filtered_img.max()) * 255
    return filtered_img.astype(np.uint8)

# Dictionary of filters
filter_functions = {
    'mean': mean_filter,
    'min': min_filter,
    'max': max_filter,
    'sobel': sobel_filter,
    'gaussian': gaussian_filter,
    'butterworth': butterworth_filter,
    'prewitt': prewitt_filter,
    'robert': robert_filter,
    'laplacian': laplacian_filter,
    'mexican_hat': mexican_hat_filter,
}

# @st.cache_data
def process_stage(_input_img, filters, merge_method):
    if not filters:
        return _input_img
    filtered_images = [filter_functions[f](_input_img) for f in filters]
    if merge_method == 'average':
        return np.mean(np.stack(filtered_images, axis=0), axis=0).astype(np.uint8)
    elif merge_method == 'max':
        return np.max(np.stack(filtered_images, axis=0), axis=0).astype(np.uint8)
    elif merge_method == 'min':
        return np.min(np.stack(filtered_images, axis=0), axis=0).astype(np.uint8)

@st.cache_data
def image_to_ascii(image, cols, scale, selected_scale, custom_scale, invert_scale):
    gscale = custom_scale if selected_scale == 'Custom' and custom_scale else custom_scales.get(selected_scale, custom_scales['Original 10-level'])
    if not gscale:
        return None  # Handle error in main
    if invert_scale:
        gscale = gscale[::-1]
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
            avg = int(get_average_l(img))
            gsval = gscale[int((avg * (len(gscale) - 1)) // 255)]
            aimg[j] += gsval
    return aimg

def get_average_l(image):
    return getAverageL(image)

def main():
    descStr = "This program converts an image into ASCII art."
    st.title(descStr)
    st.markdown("-- RockingManny")

    imgFile = st.sidebar.file_uploader("Upload a photo: ", type=["jpg", "png"])

    if imgFile is not None:
        image = Image.open(imgFile).convert('L')
        img_array = np.array(image)

        # Sidebar controls for ASCII settings
        st.sidebar.subheader("ASCII Settings")
        scale = float(st.sidebar.slider("Image Scale", 0.01, 1.00, 0.37))
        cols = int(st.sidebar.slider("Columns", 1, 1080, 700))
        zoom_level = st.sidebar.slider("Zoom", 0.01, 1.0, 0.1)
        scale_options = list(custom_scales.keys()) + ['Custom']
        selected_scale = st.sidebar.selectbox('Select ASCII Scale', scale_options, index=1)
        default_scale_value = custom_scales.get(selected_scale, '@#*+=-:,. ') if selected_scale != 'Custom' else '@#*+=-:,. '
        custom_scale = st.sidebar.text_input('Custom ASCII Scale', value=default_scale_value)
        invert_image = st.sidebar.checkbox('Invert Image?', value=True)
        invert_scale = st.sidebar.checkbox('Invert ASCII Scale?', value=False)

        # Custom Filter Pipeline GUI
        st.sidebar.subheader("Filter Pipeline")
        num_stages = st.sidebar.slider("Number of Stages", 1, 5, 1)
        pipeline = []
        for stage in range(num_stages):
            st.sidebar.subheader(f"Stage {stage+1}")
            filters = st.sidebar.multiselect(
                f"Select filters for stage {stage+1} (applied in parallel)",
                list(filter_functions.keys()),
                key=f"filters_stage_{stage}"
            )
            merge_method = st.sidebar.selectbox(
                f"Merge method for stage {stage+1}",
                ['average', 'max', 'min'],
                key=f"merge_stage_{stage}"
            )
            pipeline.append({'filters': filters, 'merge_method': merge_method})

        # Process the image through the pipeline
        original_img = img_array
        intermediate_arrays = [original_img]
        current_img = original_img
        for stage_idx, stage in enumerate(pipeline):
            st.subheader(f"Stage {stage_idx+1} Individual Filter Outputs")
            filtered_images = []
            filter_names = []
            if stage['filters']:
                for f in stage['filters']:
                    filtered_img = filter_functions[f](current_img)
                    filtered_images.append(filtered_img)
                    filter_names.append(f)
                if filtered_images:
                    st.image(filtered_images, caption=[f"Filter: {name}" for name in filter_names], width=150)
            else:
                st.write("No filters applied in this stage.")
            current_img = process_stage(current_img, stage['filters'], stage['merge_method'])
            intermediate_arrays.append(current_img)
            st.subheader(f"After Stage {stage_idx+1}")
            st.image(current_img, caption=f"After Stage {stage_idx+1}", width=150)

        # Display stage results
        st.subheader("Filter Pipeline Stage Results")
        captions = ['Original'] + [f"After Stage {i+1}" for i in range(num_stages)]
        st.image(intermediate_arrays, caption=captions, width=150)

        # ASCII art conversion
        if st.button("Convert to ASCII Art"):
            final_img_array = intermediate_arrays[-1]
            if invert_image:
                final_img_array = 255 - final_img_array
            final_image = Image.fromarray(final_img_array)
            aimg = image_to_ascii(final_image, cols, scale, selected_scale, custom_scale, invert_scale)
            
            if aimg is None:
                st.error("Custom scale cannot be empty. Using default scale.")
                aimg = image_to_ascii(final_image, cols, scale, 'Original 10-level', None, invert_scale)
            
            with st.container():
                content = '<br>'.join([str(row) for row in aimg])
                division = f"""<div class="zoom">{content}</div>"""
                st.write(f"""
                    <style>
                    .zoom {{
                        transform: scale({zoom_level});
                        transform-origin: 0 0;
                        font-family: 'Consolas', monospace;
                        color: white;
                        white-space: pre;
                    }}
                    </style>
                    {division}
                """, unsafe_allow_html=True)

            # Write to file
            outFile = 'Converted Images/out.txt'
            with open(outFile, 'w') as f:
                for row in aimg:
                    f.write(row + '\n')
            st.success(f"ASCII art written to {outFile}")
    else:
        st.warning("Please upload an image file (jpg or png)")

if __name__ == '__main__':
    main()