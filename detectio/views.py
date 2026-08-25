import base64

import cv2
import numpy as np
from django.conf import settings
from django.shortcuts import render


def home(request):
	context = {}
	if request.method == 'POST':
		uploaded_file = request.FILES.get('image')
		if not uploaded_file:
			context['error'] = 'Choose an image before starting detection.'
		elif uploaded_file.size > 10 * 1024 * 1024:
			context['error'] = 'Images must be smaller than 10 MB.'
		else:
			image_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
			image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
			if image is None:
				context['error'] = 'That file is not a readable image.'
			else:
				detector = cv2.FaceDetectorYN_create(
					str(settings.BASE_DIR / 'detectio' / 'models' / 'face_detection_yunet_2023mar.onnx'),
					'',
					(image.shape[1], image.shape[0]),
					score_threshold=0.8,
				)
				_, detections = detector.detect(image)
				detections = detections if detections is not None else []
				for detection in detections:
					x, y, width, height = map(int, detection[:4])
					cv2.rectangle(image, (x, y), (x + width, y + height), (23, 107, 104), 3)
				success, encoded_image = cv2.imencode('.jpg', image)
				if not success:
					context['error'] = 'The image could not be processed.'
				else:
					image_data = base64.b64encode(encoded_image).decode('ascii')
					context.update({
						'image_data': f'data:image/jpeg;base64,{image_data}',
						'face_count': len(detections),
					})
	return render(request, 'detectio/home.html', context)
