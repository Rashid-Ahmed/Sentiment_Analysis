    # %%
import socket
host = "127.0.0.1"  # as both code is running on same pc
port = 10000  # socket server port number
import json
client_socket = socket.socket()  # instantiate
client_socket.connect((host, port))  # connect to the server
json_data = ''
while True:
    sentence = "something (you get this from the user)"
    client_socket.send(sentence.encode())  # send message
    #if the user chat is over you set sentence to "connection closed"
    if sentence == "connection closed":
        break
    data = client_socket.recv(64000).decode()  # receive response
    data = json.loads(data)
    json_data = data
    print (data)

client_socket.close()  # close the connection


