import os 
import cv2
import aiohttp 
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def access_camera():
    api_key = os.getenv('API_KEY')


    headers = {
        "Authorization": f"Bearer {api_key}"  
    } 
    
    url = "https://api.netatmo.com/api/gethomesdata"

    async with aiohttp.ClientSession() as session:

        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                process_camera_data(data)
            else:
                print(f"Erreur: {response.status}")
                print(await response.text())
 
def process_camera_data(data):
    homes = data.get('body', {}).get('homes', [])
    for home in homes:
        for camera in home.get('cameras', []):
            print(f"Nom: {camera.get('name')}")
            print(f"VPN URL: {camera.get('vpn_url')}")
            print('---')
            face_dectector= cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            videocapture = cv2.VideoCapture(camera.get('vpn_url')+"/live/files/low/index.m3u8")
            while True:
                ret, frame = videocapture.read()
                if ret == False:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_dectector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        videocapture.release()
        cv2.destroyAllWindows()
 
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(access_camera())



    
