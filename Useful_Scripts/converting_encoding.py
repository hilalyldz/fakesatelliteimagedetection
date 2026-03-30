file_path = 'C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/resnet_log/events.out.tfevents.1737010442.EOC-001241'
output_file = 'C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/resnet_log/converted_log_file.log'

# Specify the detected or expected encoding
with open(file_path, 'r', encoding='utf-8') as f:  # Change 'utf-8' to the detected encoding if needed
    content = f.read()

print(content)
