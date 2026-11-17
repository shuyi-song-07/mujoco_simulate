"""Local readiness/stop traffic must ignore desktop and environment proxies."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from collection.cli import local_urlopen, stop_idle_backend


class LocalHTTPTests(unittest.TestCase):
    def test_health_and_idle_stop_bypass_unreachable_proxy(self):
        commands=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                body=json.dumps({'session':'scripted_test','recording':False,'saving':False}).encode()
                self.send_response(200);self.end_headers();self.wfile.write(body)
            def do_POST(self):
                commands.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200);self.end_headers();self.wfile.write(b'{"ok":true}')
        with ThreadingHTTPServer(('127.0.0.1',0),Handler) as server:
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            port=server.server_port
            try:
                with patch.dict('os.environ',{'http_proxy':'http://127.0.0.1:1',
                    'https_proxy':'http://127.0.0.1:1','no_proxy':'','NO_PROXY':''}):
                    with local_urlopen(f'http://127.0.0.1:{port}/health',timeout=1) as response:
                        self.assertEqual(json.load(response)['session'],'scripted_test')
                    self.assertTrue(stop_idle_backend(port))
                self.assertEqual(commands,[{'command':'record_stop','session':'scripted_test'}])
            finally:
                server.shutdown();worker.join()

    def test_remote_urls_cannot_use_local_proxy_bypass(self):
        with self.assertRaises(ValueError):
            local_urlopen('https://example.com',timeout=1)


if __name__=='__main__':unittest.main()
