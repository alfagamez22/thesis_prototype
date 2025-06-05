import cv2
import matplotlib.pyplot as plt

# Load the first frame from the video
cap = cv2.VideoCapture(r'H:\12HOUR_VIDEO_CCTV\thesis_prototype\backend\har\assets\rtsp.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to load video or frame.")
    exit(1)

# Mirror flip the frame horizontally
frame = cv2.flip(frame, 1)

# Convert to RGB for matplotlib
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

coords = []

fig, ax = plt.subplots()
ax.imshow(frame_rgb)
ax.set_title("Click to add dots. Close window to finish.")

line, = ax.plot([], [], 'ro-')  # red dots and lines

def onclick(event):
    if event.inaxes != ax:
        return
    x, y = int(event.xdata), int(event.ydata)
    coords.append((x, y))
    # Update plot
    xs, ys = zip(*coords)
    line.set_data(xs, ys)
    ax.annotate(f'({x},{y})', (x, y), color='yellow', fontsize=10, weight='bold')
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print("Clicked coordinates:")
for c in coords:
    print(c)
