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
            net = cv2.VideoCapture(camera.get('vpn_url')+"/live/files/low/index.m3u8")
            webcam = cv2.VideoCapture(0)
            
            if not net.isOpened():
                print("Erreur d'ouverture de la caméra réseau")
                continue

            if not webcam.isOpened():
                print("Erreur d'ouverture de la webcam")
                continue
            fps =net.get(cv2.CAP_PROP_FPS)
            fps_web =webcam.get(cv2.CAP_PROP_FPS)
            while True:
                ret, frame = net.read()
                ret_web, frame_web = webcam.read()
                if ret == False:
                    break
                if ret_web == False:    
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_web = cv2.cvtColor(frame_web, cv2.COLOR_BGR2GRAY)
                faces = face_dectector.detectMultiScale(gray, 1.3, 5)
                faces_web = face_dectector.detectMultiScale(gray_web, 1.3, 5)

                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                for (x, y, w, h) in faces_web:
                    cv2.rectangle(frame_web, (x, y), (x+w, y+h), (255, 0, 0), 2)

                
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame_web, f"FPS: {fps_web:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                height = min(frame.shape[0], frame_web.shape[0])
                frame_net_resized = cv2.resize(frame, (int(frame.shape[1] * height / frame.shape[0]), height))
                frame_web_resized = cv2.resize(frame_web, (int(frame_web.shape[1] * height / frame_web.shape[0]), height))
                combined_frame = cv2.hconcat([frame_net_resized, frame_web_resized])  
                cv2.imshow('2 camera', combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):

                    break
                    
        net.release()
        webcam.release()    
        
        cv2.destroyAllWindows()
 
async def main():
    await access_camera()

if __name__ == "__main__":
    asyncio.run(main())