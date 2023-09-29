from pylsl import StreamInlet, resolve_stream
import numpy as np

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream()
print(streams)

data = []
time_count = 0.0

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample, timestamp = inlet.pull_sample()
    data.append(sample)

    time_count += 1 * 0.004
    time_count = np.round(time_count ,3)
    print(time_count)

    if time_count == 1:
        np.asarray(data)
        print(timestamp, data, np.shape(data))
        break
