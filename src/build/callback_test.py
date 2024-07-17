import napari
import numpy as np

# Sample 3D image and labels data
image_data = np.random.random((100, 100, 100))
labels_data = np.zeros((100, 100, 100), dtype=int)

# Create a Napari viewer and add the labels layer
viewer = napari.Viewer()
labels_layer = viewer.add_labels(labels_data, name='labels')

# Define the callback function
def on_labels_change(event):
    print("Labels layer changed")
    print("Change details:", event)

# Connect the callback function to multiple events
labels_layer.events.set_data.connect(on_labels_change)
# labels_layer.events.changed.connect(on_labels_change)

print("Event connected")

# Modify the labels data to trigger the event
labels_layer.data[0, 0, 0] = 1

# Run the Napari event loop
napari.run()