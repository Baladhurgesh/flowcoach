import requests
import json
# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
api_url = #'https://fr9ayvbcby.loclx.io'#'http://localhost:8000/generate-text'  # Replace with your API URL

# Create a Python dictionary
payload_dict = {
    "pose": [{"x":0.28606635,"y":0.26351947,"z":-0.21968713},{"x":0.27188313,"y":0.25380006,"z":-0.19726977},{"x":0.26998338,"y":0.254031,"z":-0.19748342},{"x":0.2676681,"y":0.2543769,"z":-0.19770557},{"x":0.27007526,"y":0.2549526,"z":-0.26008326},{"x":0.26724133,"y":0.2559356,"z":-0.26010635},{"x":0.26401824,"y":0.2570626,"z":-0.2601546},{"x":0.24785459,"y":0.266294,"z":-0.07431645},{"x":0.24263772,"y":0.26918644,"z":-0.36116007},{"x":0.28586888,"y":0.27897623,"z":-0.16781315},{"x":0.28169256,"y":0.28003022,"z":-0.25162455},{"x":0.22686844,"y":0.3323521,"z":0.107659176},{"x":0.2344546,"y":0.3329187,"z":-0.4689553},{"x":0.31214193,"y":0.37014407,"z":0.25239044},{"x":0.37242234,"y":0.34090036,"z":-0.56488675},{"x":0.40561688,"y":0.38519636,"z":0.20894942},{"x":0.48947817,"y":0.29553276,"z":-0.5198781},{"x":0.42213666,"y":0.38363585,"z":0.21708192},{"x":0.52583843,"y":0.28678864,"z":-0.5951515},{"x":0.42645797,"y":0.37814355,"z":0.15890528},{"x":0.52154064,"y":0.27607843,"z":-0.56165487},{"x":0.42107132,"y":0.3808329,"z":0.18201321},{"x":0.5082439,"y":0.2796815,"z":-0.50815445},{"x":0.25792855,"y":0.51053935,"z":0.15313983},{"x":0.23266378,"y":0.52281356,"z":-0.153159},{"x":0.41810513,"y":0.5163142,"z":0.31679577},{"x":0.16933025,"y":0.6266074,"z":-0.020530565},{"x":0.42308402,"y":0.6405989,"z":0.53541833},{"x":0.063442394,"y":0.63214725,"z":0.3843174},{"x":0.4051504,"y":0.6679194,"z":0.5573738},{"x":0.042692944,"y":0.6252571,"z":0.42974705},{"x":0.4990361,"y":0.6593405,"z":0.52064526},{"x":0.061396986,"y":0.65112543,"z":0.40980417}]

}# Convert the dictionary to JSON format
json_payload = json.dumps(payload_dict)

# Send the POST request with the JSON payload
response = requests.post(api_url, data=json_payload, headers={"Content-Type": "application/json"})

if response.status_code == 200:
    print("Response:")
    print(response.text)
else:
    print(f"Error: {response.status_code}\n{response.text}")