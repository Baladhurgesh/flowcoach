# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
from fastapi import FastAPI, HTTPException, Body
import httpx
import numpy as np
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarkerResult
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark, Landmark
import openai
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions'
app = FastAPI()

mapping = {0:'nose'
,1 :'left eye (inner)'
,2 :'left eye'
,3 :'left eye (outer)'
,4 :'right eye (inner)'
,5 :'right eye'
,6 :'right eye (outer)'
,7 :'left ear'
,8 :'right ear'
,9 :'mouth (left)'
,10 :'mouth (right)'
,11 :'left shoulder'
,12 :'right shoulder'
,13 :'left elbow'
,14 :'right elbow'
,15 :'left wrist'
,16 :'right wrist'
,17 :'left pinky'
,18 :'right pinky'
,19 :'left index'
,20 :'right index'
,21 :'left thumb'
,22 :'right thumb'
,23 :'left hip'
,24 :'right hip'
,25 :'left knee'
,26 :'right knee'
,27 :'left ankle'
,28 :'right ankle'
,29 :'left heel'
,30 :'right heel'
,31 :'left foot index'
,32 :'right foot index'}

async def call_openai_api(prompt: str): 
    messages = [ {"role": "user", "content": prompt} ]
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
      
    reply = chat.choices[0].message.content
    
    prompt = reply + ' ' + 'Just provide highlight on a higher level as you are instructing a user who is doing yoga'
    messages = [ {"role": "system", "content": prompt} ]
    chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
        )
    reply = chat.choices[0].message.content
    print(reply)
    return reply
def compare_pose(user_pose):
    differences = []
    x_differences = []
    y_differences = []
    diff_dict = {}
    instructor_pose = [[NormalizedLandmark(x=0.4598863124847412, y=0.3047647476196289, z=-0.1037423387169838, visibility=0.9973911046981812, presence=0.9995338916778564),
  NormalizedLandmark(x=0.44836151599884033, y=0.31052422523498535, z=-0.0871010273694992, visibility=0.9975414276123047, presence=0.9997709393501282),
  NormalizedLandmark(x=0.44790101051330566, y=0.3128851056098938, z=-0.08740547299385071, visibility=0.996432900428772, presence=0.9998213648796082),
  NormalizedLandmark(x=0.447555273771286, y=0.3154524564743042, z=-0.08733800798654556, visibility=0.9978433847427368, presence=0.9997828602790833),
  NormalizedLandmark(x=0.44803887605667114, y=0.31173405051231384, z=-0.12126023322343826, visibility=0.9982129335403442, presence=0.9996554851531982),
  NormalizedLandmark(x=0.44753462076187134, y=0.31463658809661865, z=-0.12166554480791092, visibility=0.9983951449394226, presence=0.9997343420982361),
  NormalizedLandmark(x=0.44716876745224, y=0.31755906343460083, z=-0.12189547717571259, visibility=0.9988382458686829, presence=0.9996949434280396),
  NormalizedLandmark(x=0.4457078278064728, y=0.3400730788707733, z=-0.009492205455899239, visibility=0.9987397789955139, presence=0.999636173248291),
  NormalizedLandmark(x=0.44611185789108276, y=0.34113502502441406, z=-0.16602744162082672, visibility=0.9985278844833374, presence=0.9994617104530334),
  NormalizedLandmark(x=0.46792367100715637, y=0.31779342889785767, z=-0.07101862132549286, visibility=0.9965533018112183, presence=0.9996383190155029),
  NormalizedLandmark(x=0.4673919975757599, y=0.3193747103214264, z=-0.11888846755027771, visibility=0.9966363906860352, presence=0.9992712140083313),
  NormalizedLandmark(x=0.47640174627304077, y=0.4063907563686371, z=0.0788835808634758, visibility=0.9994812607765198, presence=0.9999247789382935),
  NormalizedLandmark(x=0.4685068130493164, y=0.4057638943195343, z=-0.25438109040260315, visibility=0.9998549222946167, presence=0.9998817443847656),
  NormalizedLandmark(x=0.4923988878726959, y=0.2652871906757355, z=0.1054176539182663, visibility=0.047820065170526505, presence=0.9999730587005615),
  NormalizedLandmark(x=0.48935773968696594, y=0.2511882185935974, z=-0.3754613697528839, visibility=0.9926277995109558, presence=0.9998984336853027),
  NormalizedLandmark(x=0.4914699196815491, y=0.15204459428787231, z=0.015510890632867813, visibility=0.17718356847763062, presence=0.9998465776443481),
  NormalizedLandmark(x=0.4884490370750427, y=0.1299780011177063, z=-0.4147598147392273, visibility=0.9905975461006165, presence=0.999771773815155),
  NormalizedLandmark(x=0.4902445077896118, y=0.11768651008605957, z=0.004022460896521807, visibility=0.2879154086112976, presence=0.9996261596679688),
  NormalizedLandmark(x=0.4873717427253723, y=0.09449005126953125, z=-0.46810051798820496, visibility=0.9865341186523438, presence=0.9995361566543579),
  NormalizedLandmark(x=0.484273225069046, y=0.11851513385772705, z=-0.023607848212122917, visibility=0.28189629316329956, presence=0.9995489716529846),
  NormalizedLandmark(x=0.4812436103820801, y=0.0962371826171875, z=-0.45139873027801514, visibility=0.9841412901878357, presence=0.9993000030517578),
  NormalizedLandmark(x=0.4841839671134949, y=0.12764602899551392, z=-0.0007263199076987803, visibility=0.3019046485424042, presence=0.9996979236602783),
  NormalizedLandmark(x=0.4812529683113098, y=0.10868507623672485, z=-0.4137445390224457, visibility=0.9736766815185547, presence=0.9996616840362549),
  NormalizedLandmark(x=0.5126389265060425, y=0.678322970867157, z=0.10280706733465195, visibility=0.999994158744812, presence=0.9999983310699463),
  NormalizedLandmark(x=0.48405909538269043, y=0.6870208382606506, z=-0.10314568877220154, visibility=0.9999887943267822, presence=0.9999977350234985),
  NormalizedLandmark(x=0.6055759787559509, y=0.7089958786964417, z=0.1682724505662918, visibility=0.9991814494132996, presence=0.9999276399612427),
  NormalizedLandmark(x=0.3967733085155487, y=0.8001656532287598, z=-0.15130892395973206, visibility=0.9991503953933716, presence=0.9998911619186401),
  NormalizedLandmark(x=0.6077895164489746, y=0.8958825469017029, z=0.2912173569202423, visibility=0.9998328685760498, presence=0.9999293088912964),
  NormalizedLandmark(x=0.29410749673843384, y=0.8872659206390381, z=-0.038915004581213, visibility=0.9986961483955383, presence=0.999744713306427),
  NormalizedLandmark(x=0.5940995216369629, y=0.9320518970489502, z=0.2972945272922516, visibility=0.9997009038925171, presence=0.9998705387115479),
  NormalizedLandmark(x=0.2714473605155945, y=0.8968328237533569, z=-0.02808176353573799, visibility=0.9917669892311096, presence=0.9997841715812683),
  NormalizedLandmark(x=0.6521375179290771, y=0.9435231685638428, z=0.23466233909130096, visibility=0.9992833733558655, presence=0.9997496008872986),
  NormalizedLandmark(x=0.3073768615722656, y=0.9609691500663757, z=-0.10913737863302231, visibility=0.9967566132545471, presence=0.9996856451034546)]]
    tolerance_threshold = 0.1
    # Iterate through each landmark
    for i in range(len(instructor_pose[0])):
        user_landmark = np.array([user_pose[i]['x'], user_pose[i]['y'], user_pose[i]['z']])
        instructor_landmark = np.array([instructor_pose[0][i].x, instructor_pose[0][i].y, instructor_pose[0][i].z])
        user_landmark_tuple = (user_pose[i]['x'], user_pose[i]['y'])
        instructor_landmark_tuple = (instructor_pose[0][i].x, instructor_pose[0][i].y)
        # Calculate the Euclidean distance between user and instructor landmarks
        distance = np.linalg.norm(user_landmark - instructor_landmark)

        # Iterate through the pose points and find the maximum differences in X and Y

        diff_x = abs(user_landmark_tuple[0] - instructor_landmark_tuple[0])
        diff_y = abs(user_landmark_tuple[1] - instructor_landmark_tuple[1])

        x_differences.append(diff_x)
        y_differences.append(diff_y)
        # Append the difference to the list
        differences.append(distance)

        landmark = mapping[i]
        diff_dict[landmark] = {'x':diff_x,'y':diff_y,'xy': distance}
        
    rem_list = ['left eye (inner)','left eye','left eye (outer)','right eye (inner)','right eye',\
                'right eye (outer)','left ear','right ear','mouth (left)','mouth (right)', 'left pinky'\
                    ,'right pinky','left index','right index','left thumb','right thumb', 'left heel', 'right heel']
    [diff_dict.pop(key) for key in rem_list]
    diff_dict['head'] = diff_dict['nose']
    del diff_dict['nose']
    # print(diff_dict)
    return diff_dict
@app.post("/generate-text")
async def generate_text(prompt: dict = Body(...)):
    try:
        # print(prompt)
        # print(prompt['pose'])
        response_text = True
        deviation = compare_pose(prompt['pose'])
        prompt_text = "Assume you are certified yoga instructor. User is trying to do the crescent yoga pose. You should help the user achieve the correct pose, so that he won't injure himself. \
            You are given a dictionary containing body part mapped to deviations in x, y and x-y wrt correct pose.\
            Please provide suggestions for the body parts with the most deviations from correct pose. Include the top 3 & only include the ones where a significant change is required. \
            Provide suggestions as you would to a student in the middle of a yoga session."
        final_prompt = prompt_text + ' ' + str(deviation)
        print(final_prompt)
        print ('')
        print ('')
        response_text = await call_openai_api(final_prompt)
        return response_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))