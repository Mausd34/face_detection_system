from io import BytesIO

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image


class HomeViewTests(TestCase):
	def test_homepage_loads(self):
		response = self.client.get('/')

		self.assertEqual(response.status_code, 200)
		self.assertContains(response, 'Face Detection Studio')

	def test_upload_requires_an_image(self):
		response = self.client.post('/')

		self.assertContains(response, 'Choose an image before starting detection.')

	def test_valid_image_is_processed(self):
		image_stream = BytesIO()
		Image.new('RGB', (100, 100), color='white').save(image_stream, format='JPEG')
		upload = SimpleUploadedFile(
			'sample.jpg', image_stream.getvalue(), content_type='image/jpeg'
		)

		response = self.client.post('/', {'image': upload})

		self.assertEqual(response.status_code, 200)
		self.assertEqual(response.context['face_count'], 0)
		self.assertTrue(response.context['image_data'].startswith('data:image/jpeg;base64,'))
