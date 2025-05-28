import socket 
import pickle
from get_drawing_touch import get_2d_demos
from get_drawing_rt import get_2d_demos_rt
import cv2

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 8080)
s.bind(server_address)
print('Starting up on {} port {}'.format(*server_address))
s.listen()
print("listening")
conn, addr = s.accept()
print("Connection Established")

# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# server_address = ('172.16.0.3', 10980)
# sock.connect(server_address)
# print("Connection to Host Established")

while True:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = ('172.16.0.3', 10980)
        sock.connect(server_address)

        sock.settimeout(1000)
        rec = sock.recv(int(1e8))
        img = b''
        sock.settimeout(1)
        while rec != b'':
            try:
                img += rec
                rec = sock.recv(int(1e8))
            except socket.timeout:
                break
        print("image recieved")
        data = pickle.loads(img)
        camera = data[0]
        img = data[1]
        task = data[2]
        if len(data)==4:
            alg = data[3]
        else:
            alg = 'common'
        cv2.imwrite("robot_img.png", img)
        print(alg)
        if alg == 'rt_traj':
            traj, heights = get_2d_demos_rt(conn, camera, task)
            print(len(traj)) 
            data = pickle.dumps([traj, heights])
        else:
            traj = get_2d_demos(conn, camera, task)
            print(len(traj)) 
            data = pickle.dumps(traj)
        if traj  == "done":
            sock.sendall(pickle.dumps("done"))
            exit()
        print(len(data))
        print(len(pickle.loads(data)))

        sock.sendall(data)
    except socket.timeout or KeyboardInterrupt:
        print("!!!!!!")
        sock.sendall(pickle.dumps("done"))
        continue
    
    