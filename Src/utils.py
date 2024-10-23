import matplotlib.pyplot as plt

def display_prediction(img, class_name):
    plt.imshow(img)
    plt.title(f'Identified as {class_name}')
    plt.axis('off')
    plt.show()
