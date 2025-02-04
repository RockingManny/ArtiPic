# Python code to convert an image to ASCII image.
import numpy as np
from PIL import Image
import streamlit as st
 
# gray scale level values from: 
# http://paulbourke.net/dataformats/asciiart/
 
# 70 levels of gray
gscale1 = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1[]?-_+~i!lI;:,^`'. "

# 10 levels of gray
gscale2 = '@%#*+=-:. '

# calculates the average index of a pixel: used to select appropriate gscale ascii character
def getAverageL(image):
 
    """
    Given PIL Image, return average value of grayscale value
    """

    # get image as numpy array
    im = np.array(image)

    # get shape
    w,h = im.shape

    # get average
    return np.average(im.reshape(w*h))

# Convert img to ASCII art: returns list of rows containing ascii art
def covertImageToAscii(fileName, cols, scale, moreLevels):
    
    """
    Given Image and dims (rows, cols) returns an m*n list of Images 
    """

    # declare globals
    global gscale1, gscale2
 
    # open image and convert to grayscale
    image = Image.open(fileName).convert('L')
    
    # Inverts image to negative
    img_array = np.array(image)
    img_array_inverted = 255 - img_array
    image = Image.fromarray(img_array_inverted)


    # store dimensions
    W, H = image.size[0], image.size[1]
    print("input image dims: %d x %d" % (W, H))

    # compute width of tile
    w = W/cols
 
    # compute tile height based on aspect ratio and scale
    h = w/scale

    # check if image size is too small
    if cols > W or int(H/h) > H:
        print("Image too small for specified cols!")
        w=W
        h=H

    # compute number of rows
    rows = int(H/h)
    print("cols: %d, rows: %d, scale: %d" % (cols, rows, scale))
    print("tile dims: %d x %d" % (w, h))

    # ascii image is a list of character strings
    aimg = []

    # generate list of dimensions
    for j in range(rows):
        y1 = int(j*h)
        y2 = int((j+1)*h)
 
        # correct last tile
        if j == rows-1:
            y2 = H

        # append an empty string
        aimg.append("")
        for i in range(cols):

            # crop image to tile
            x1 = int(i*w)
            x2 = int((i+1)*w)

            # correct last tile
            if i == cols-1:
                x2 = W

            # crop image to extract tile
            img = image.crop((x1, y1, x2, y2))
 
            # get average luminance
            avg = int(getAverageL(img))

            # look up ascii char
            if moreLevels:
                gsval = gscale1[int((avg*(len(gscale1)-1))/255)]
            else:
                gsval = gscale2[int((avg*(len(gscale2)-1))/255)]

            # append ascii char to string
            aimg[j] += gsval

    # return txt image
    return aimg

# main() function streamlit
def main():

    # create parser
    descStr = "This program converts an image into ASCII art."

    st.title(descStr)
    st.markdown("-- RockingManny")

    imgFile = st.file_uploader("Choose a photo: ",type="jpg")

    if imgFile is not None:
        morelevels = False
        scale = 0.5
        cols = 500

        scale = (float)(st.sidebar.slider("image scale", 0.01,1.00,0.01))
        cols = (int)(st.sidebar.slider("cols", 1,1080,1))
        morelevels = st.sidebar.checkbox('More Levels?')

        print('generating ASCII art...')
        # convert image to ascii txt
        aimg = covertImageToAscii(imgFile, cols, scale, morelevels)
        
        with st.container():
            
            zoom_level = st.sidebar.slider("Zoom", 0.01, 1.0, 0.01)
            content = '<br>'.join([str(row) for row in aimg])
            division=f"""<div class="zoom">+{content}+</div>"""
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
                
        print("Image Generated!")

        # set output file
        outFile = 'Converted Images\out.txt'
 
        # open file
        f = open(outFile, 'w')
    
        # write to file
        for row in aimg:
            f.write(row + '\n')
    
        # cleanup
        f.close()
        
        # Open the output file
        print("ASCII art written to %s" % outFile)
        # subprocess.call(["notepad.exe", outFile])

    else:
        st.warning("Please choose a file to upload")

# call main
if __name__ == '__main__':
    main()