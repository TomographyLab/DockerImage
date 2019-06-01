# ilang - Inference Language
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 


import _thread as thread
import http.server as BaseHTTPServer
import socketserver as SocketServer

host = '0.0.0.0'
port = 8080

def serve(host,port):
    handler = BaseHTTPServer.SimpleHTTPRequestHandler
    SocketServer.TCPServer.allow_reuse_address = True
    server = SocketServer.TCPServer((host, port), handler, bind_and_activate=False)
    server.allow_reuse_address=True
    try:
        server.server_bind()
        server.server_activate()
        print("serving at port:" + str(port))
        server.serve_forever()
    except:
        server.server_close()

def run_webserver(background):
    try:
        if background:
            thread.start_new_thread(serve, (host,port))
        else:
            serve(host,port)
    except:
        print('server already running')


