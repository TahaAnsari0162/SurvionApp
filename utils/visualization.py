import matplotlib
matplotlib.use('Agg')  # Use the Agg backend

import matplotlib.pyplot as plt
import cv2
import numpy as np
import io
import base64

def display_image_grid(images, titles=None, rows=None, cols=4, figsize=(20, 5)):
    """Displays a grid of images and return their base64 representation."""
    num_images = len(images)
    if rows is None:
        rows = int(np.ceil(num_images / cols))
    fig = plt.figure(figsize=(figsize[0], figsize[1] * rows))
    image_base64_list = []
    for idx, image in enumerate(images):
      plt.subplot(rows, cols, idx + 1)
      if isinstance(image, str):
          img = cv2.imread(image)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      else:
          img = image
      if isinstance(img, np.ndarray):  # Check if image is a NumPy array
           plt.imshow(img)
      else:
         plt.imshow(img)
      if titles and idx < len(titles):
          plt.title(titles[idx])
      plt.axis('off')
    plt.tight_layout()

    for i in range(num_images):
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        image_base64_list.append(image_base64)
    plt.close(fig)
    return image_base64_list
def create_bar_chart(actions, scores):
    """Creates a bar chart and returns its base64 representation."""
    colors = ['red' if score == max(scores) else 'blue' for score in scores]
    fig = plt.figure(figsize=(10, 4))
    plt.barh(actions, scores, color=colors)
    plt.title('Action Probability Scores')
    plt.xlabel('Confidence Score')
    plt.xlim(0, 1)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64
def create_pie_chart(labels, sizes):
    """Creates a pie chart and returns its base64 representation."""
    colors = ['lightcoral', 'lightblue', 'lightgreen']
    fig = plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%',
             colors=colors, explode=(0.1, 0, 0))
    plt.title('Distribution of Stealing Scene Types')
    plt.axis('equal')
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return image_base64
def create_barplot(x_values, y_values, title, x_label, y_label):
  """Creates a bar plot and returns it's base64 representation"""
  fig = plt.figure(figsize=(12, 6))
  plt.bar(x_values, y_values, color='skyblue')
  plt.title(title, pad=20)
  plt.xticks(rotation=45)
  plt.ylabel(y_label)
  for i, v in enumerate(y_values):
      plt.text(i, v + 0.5, str(v), ha='center')
  plt.tight_layout()
  buffer = io.BytesIO()
  fig.savefig(buffer, format='png', bbox_inches='tight')
  buffer.seek(0)
  image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
  plt.close(fig)
  return image_base64