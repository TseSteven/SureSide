
# Example of embedding CEF Python browser using wxPython library.
# This example has a top menu and a browser widget without navigation bar.

# Tested configurations:
# - wxPython 4.0 on Windows/Mac/Linux
# - wxPython 3.0 on Windows/Mac
# - wxPython 2.8 on Linux
# - CEF Python v66.0+

import wx
import wx.adv
from wx.lib.pubsub import pub
from cefpython3 import cefpython as cef
import platform
import sys
import os
import configparser
from kombu import Connection, Exchange, Queue, binding
from kombu.mixins import ConsumerMixin
import traceback
import time
import threading
from urllib.request import urlretrieve
import json
import random

icon_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),"icon.ico")

class Worker(ConsumerMixin):
    def __init__(self, connection, queues, print_enable):
        self.connection = connection
        self.queues = queues
        self.print_enable = print_enable
        print('Listening.....')

    def get_consumers(self, Consumer, channel):
        return [Consumer(queues=self.queues,
                         callbacks=[self.on_message])]

    def on_message(self, body, message):
        print('Got message: {0}'.format(body))
        data = body
        message.ack()
        if 'web_url' in data:
            web_url = data['web_url']
            print('Web URL: '+ web_url)
            wx.CallAfter(pub.sendMessage, "msg_update", msg=web_url)
        if 'pdf_url' in data and 'msg_id' in data:
            msg_id = data['msg_id']
            pdf_url = data['pdf_url']
            print('PDF URL: '+ pdf_url)
            
            if self.print_enable == 'yes':
                try:
                    filename = msg_id + '.pdf'
                    pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"pdf")
                    file_path = os.path.join(pdf_path, filename)
                    urlretrieve(pdf_url, file_path)
                    time.sleep(3)
                    os.startfile(file_path, "print")
                except:
                    print(traceback.format_exc())
                    pass
        

class Kombu_Receive_Thread(threading.Thread):
    def __init__(self, settings_dict):
        threading.Thread.__init__(self)
        self.settings_dict = settings_dict
        self.channel = None
    
    def run(self):
        #if self.stop_flag == True:
            #break
        server_ip = self.settings_dict['server_ip']
        os.environ['NO_PROXY'] = server_ip
        
        topic = self.settings_dict['topic']
        print_enable = self.settings_dict['print_enable']
        
        exchange = Exchange("warning", type="topic")

        if topic == '#':
            binding_keys = '#'
        elif  ',' in topic:
            binding_keys = topic.split(',')
        elif '.' in topic:
            binding_keys = [topic]

        binding_list = []

        for binding_key in binding_keys:
            binding_list.append(binding(exchange, routing_key=binding_key.strip()))
        
       
        queues = [Queue('', exchange=exchange, bindings=binding_list)]
        
       
        if ',' in server_ip:
            ip_list = server_ip.split(',')
        else:
            ip_list = [server_ip]
        
        print(ip_list)
        
        primary_ip = ip_list[0].strip()
        alternates_ip = []
        if len(ip_list)>1:
            for item in ip_list[1:]:
                alternates_ip.append('amqp://rad:rad@{}:5672//'.format(item.strip()))
        
        with Connection('amqp://rad:rad@{}:5672//'.format(primary_ip), alternates=alternates_ip, failover_strategy='round-robin', heartbeat=4) as conn:
            try:
                self.worker = Worker(conn, queues, print_enable)
                self.worker.run()
            except:
                print(traceback.format_exc())
                pass
                

    def stop(self):
        print('kombu thread stopped!')
        self.worker.should_stop = True
        self.join()

def create_menu_item(menu, label, func):
    item = wx.MenuItem(menu, -1, label)
    menu.Bind(wx.EVT_MENU, func, id=item.GetId())
    menu.Append(item)
    return item

class SettingDialog(wx.Dialog): 
    def __init__(self, parent, title, settings_dict): 
        super(SettingDialog, self).__init__(parent, title = title,size = (400,400))

        panel = wx.Panel(self) 
        vbox = wx.BoxSizer(wx.VERTICAL) 
         
        hbox0 = wx.BoxSizer(wx.HORIZONTAL) 
        label_server_ip = wx.StaticText(panel, -1, "Server IP Address") 
        hbox0.Add(label_server_ip, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
        self.txt_ctrl_server_ip = wx.TextCtrl(panel, size=(250, -1)) 
        self.txt_ctrl_server_ip.SetValue(settings_dict['server_ip'])
        hbox0.Add(self.txt_ctrl_server_ip,1,wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL,5) 
        vbox.Add(hbox0) 

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        lblList = ['Yes', 'No']     
        self.rbox = wx.RadioBox(panel,label = 'Enable Printing?', pos = (80,10), size=(400, -1), choices = lblList , majorDimension = 1, style = wx.RA_SPECIFY_ROWS) 
        if settings_dict['print_enable'] == 'yes':
            self.rbox.SetSelection(0)
        else:
            self.rbox.SetSelection(1)
        hbox1.Add(self.rbox, 1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
        vbox.Add(hbox1) 

        hbox2 = wx.BoxSizer(wx.HORIZONTAL) 
        label_topic = wx.StaticText(panel, -1, "Message Queue Topics") 
        hbox2.Add(label_topic,1, wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
        self.txt_ctrl_topic = wx.TextCtrl(panel,size = (250,200),style = wx.TE_MULTILINE) 
        self.txt_ctrl_topic.SetValue(settings_dict['topic'])
        hbox2.Add(self.txt_ctrl_topic,1,wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL,5) 
        vbox.Add(hbox2)
        
        vbox.AddStretchSpacer(2)

        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.ok_btn = wx.Button(panel, -1, "OK")
        self.cancel_btn = wx.Button(panel, -1, "Cancel")
        hbox3.AddStretchSpacer(4) 
        hbox3.Add(self.ok_btn,1,wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL,5) 
        hbox3.Add(self.cancel_btn,1,wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL,5) 
        vbox.Add(hbox3)
        panel.SetSizer(vbox) 
        self.Centre()
        self.setup_icon()
        self.Show() 
        self.Fit()  

        self.ok_btn.Bind(wx.EVT_BUTTON,self.OnOKClicked) 
        self.cancel_btn.Bind(wx.EVT_BUTTON,self.OnCancelClicked) 

    def setup_icon(self):
        icon = wx.Icon()
        icon.CopyFromBitmap(wx.Bitmap(icon_file, wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)

    def OnOKClicked(self,event): 
        
        parser = configparser.ConfigParser()
        parser.read('setting.ini')
        
        server_ip = self.txt_ctrl_server_ip.GetValue().replace(' ', '')
        topic = self.txt_ctrl_topic.GetValue().replace(' ', '')
        
        parser.set('Networking', 'server_ip', server_ip)
        parser.set('Networking', 'topic', topic)
        
        if self.rbox.GetSelection()==0:
            parser.set('Networking', 'print_enable', 'yes')
        else:
            parser.set('Networking', 'print_enable', 'no')
            
        with open('setting.ini', 'w') as configfile:    # save
            parser.write(configfile)
        wx.MessageBox('Please restart application to enable new settings!', 'Info', wx.OK | wx.ICON_INFORMATION)
        self.EndModal(wx.ID_OK)
        wx.CallAfter(self.Destroy)
        
    def OnCancelClicked(self,event): 
        self.EndModal(wx.ID_CANCEL)
        wx.CallAfter(self.Destroy)
        
    def GetSettings(self):
        return self.settings

class TaskBarIcon(wx.adv.TaskBarIcon):
    def __init__(self, frame):
        self.frame = frame
        super(TaskBarIcon, self).__init__()
        self.set_icon(icon_file)
        self.Bind(wx.adv.EVT_TASKBAR_LEFT_DOWN, self.on_left_down)
        self.settings_dict = self.read_config()
        self.kombu_thread = Kombu_Receive_Thread(self.settings_dict)
        self.kombu_thread.start()

    def read_config(self):
        parser = configparser.ConfigParser()
        parser.read('setting.ini')        
        if parser.has_section('Networking') == False:
            parser.add_section('Networking')            
        if parser.has_option('Networking','server_ip') == False:
            parser.set('Networking', 'server_ip', '10.0.2.15')
        if parser.has_option('Networking','print_enable') == False:
            parser.set('Networking', 'print_enable', 'no')
        if parser.has_option('Networking','topic') == False:
            parser.set('Networking', 'topic', 'POHCT.delay_upload')
        with open('setting.ini', 'w') as configfile: 
            parser.write(configfile)        

        parser.read('setting.ini')
        server_ip = parser.get('Networking','server_ip')
        topic = parser.get('Networking', 'topic')
        print_enable = parser.get('Networking','print_enable')
        settings_dict = {'server_ip':server_ip, 'topic':topic, 'print_enable':print_enable}
        return settings_dict
        

    def CreatePopupMenu(self):
        menu = wx.Menu()
        create_menu_item(menu, 'Version. 20191021', None)
        menu.AppendSeparator()
        create_menu_item(menu, 'Settings', self.on_setting)
        menu.AppendSeparator()
        create_menu_item(menu, 'Exit', self.on_exit)
        return menu

    def set_icon(self, path):
        icon = wx.Icon(path)
        self.SetIcon(icon, 'NTWC DR Warning')

    def on_left_down(self, event):      
        print ('Tray icon was left-clicked.')

    def on_setting(self, event):
        setting_dlg = SettingDialog(None,'Application settings', self.settings_dict)
        setting_dlg.ShowModal()

    def on_exit(self, event):
        wx.CallAfter(self.Destroy)
        self.frame.Close()
        self.kombu_thread.stop()



# class App(wx.App):
    # def OnInit(self):
        # frame=wx.Frame(None)
        # self.SetTopWindow(frame)
        # TaskBarIcon(frame)
        # return True



# Platforms
WINDOWS = (platform.system() == "Windows")
LINUX = (platform.system() == "Linux")
MAC = (platform.system() == "Darwin")

if MAC:
    try:
        # noinspection PyUnresolvedReferences
        from AppKit import NSApp
    except ImportError:
        print("[wxpython.py] Error: PyObjC package is missing, "
              "cannot fix Issue #371")
        print("[wxpython.py] To install PyObjC type: "
              "pip install -U pyobjc")
        sys.exit(1)

# Configuration
WIDTH = 900
HEIGHT = 640

# Globals
g_count_windows = 0


def main():
    pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"pdf")
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
    check_versions()
    sys.excepthook = cef.ExceptHook  # To shutdown all CEF processes on error
    settings = {}
    switches = {"proxy-server": "direct://",}
    if MAC:
        # Issue #442 requires enabling message pump on Mac
        # and calling message loop work in a timer both at
        # the same time. This is an incorrect approach
        # and only a temporary fix.
        settings["external_message_pump"] = True
    if WINDOWS:
        # noinspection PyUnresolvedReferences, PyArgumentList
        cef.DpiAware.EnableHighDpiSupport()
    cef.Initialize(settings=settings, switches=switches)
    app = CefApp(False)
    app.MainLoop()
    del app  # Must destroy before calling Shutdown
    if not MAC:
        # On Mac shutdown is called in OnClose
        cef.Shutdown()


def check_versions():
    print("[wxpython.py] CEF Python {ver}".format(ver=cef.__version__))
    print("[wxpython.py] Python {ver} {arch}".format(
            ver=platform.python_version(), arch=platform.architecture()[0]))
    print("[wxpython.py] wxPython {ver}".format(ver=wx.version()))
    # CEF Python version requirement
    assert cef.__version__ >= "66.0", "CEF Python v66.0+ required to run this"


def scale_window_size_for_high_dpi(width, height):
    """Scale window size for high DPI devices. This func can be
    called on all operating systems, but scales only for Windows.
    If scaled value is bigger than the work area on the display
    then it will be reduced."""
    if not WINDOWS:
        return width, height
    (_, _, max_width, max_height) = wx.GetClientDisplayRect().Get()
    # noinspection PyUnresolvedReferences
    (width, height) = cef.DpiAware.Scale((width, height))
    if width > max_width:
        width = max_width
    if height > max_height:
        height = max_height
    return width, height


class MainFrame(wx.Frame):

    def __init__(self, url):
        self.browser = None

        # Must ignore X11 errors like 'BadWindow' and others by
        # installing X11 error handlers. This must be done after
        # wx was intialized.
        if LINUX:
            cef.WindowUtils.InstallX11ErrorHandlers()

        global g_count_windows
        g_count_windows += 1

        if WINDOWS:
            # noinspection PyUnresolvedReferences, PyArgumentList
            print("[wxpython.py] System DPI settings: %s"
                  % str(cef.DpiAware.GetSystemDpi()))
        if hasattr(wx, "GetDisplayPPI"):
            print("[wxpython.py] wx.GetDisplayPPI = %s" % wx.GetDisplayPPI())
        print("[wxpython.py] wx.GetDisplaySize = %s" % wx.GetDisplaySize())

        print("[wxpython.py] MainFrame declared size: %s"
              % str((WIDTH, HEIGHT)))
        size = scale_window_size_for_high_dpi(WIDTH, HEIGHT)
        print("[wxpython.py] MainFrame DPI scaled size: %s" % str(size))

        wx.Frame.__init__(self, parent=None, id=wx.ID_ANY,
                          title='NTWC DR Warning!', size=size)
        # wxPython will set a smaller size when it is bigger
        # than desktop size.
        print("[wxpython.py] MainFrame actual size: %s" % self.GetSize())

        self.setup_icon()
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        # Set wx.WANTS_CHARS style for the keyboard to work.
        # This style also needs to be set for all parent controls.
        self.browser_panel = wx.Panel(self, style=wx.WANTS_CHARS)
        self.browser_panel.Bind(wx.EVT_SET_FOCUS, self.OnSetFocus)
        self.browser_panel.Bind(wx.EVT_SIZE, self.OnSize)

        if MAC:
            # Make the content view for the window have a layer.
            # This will make all sub-views have layers. This is
            # necessary to ensure correct layer ordering of all
            # child views and their layers. This fixes Window
            # glitchiness during initial loading on Mac (Issue #371).
            NSApp.windows()[0].contentView().setWantsLayer_(True)

        if LINUX:
            # On Linux must show before embedding browser, so that handle
            # is available (Issue #347).
            self.Show()
            # In wxPython 3.0 and wxPython 4.0 on Linux handle is
            # still not yet available, so must delay embedding browser
            # (Issue #349).
            if wx.version().startswith("3.") or wx.version().startswith("4."):
                wx.CallLater(100, self.embed_browser, url)
            else:
                # This works fine in wxPython 2.8 on Linux
                self.embed_browser(url)
        else:
            self.embed_browser(url)
            self.SetWindowStyle(wx.DEFAULT_FRAME_STYLE|wx.STAY_ON_TOP)
            self.Show()
            self.Maximize(True)

    def setup_icon(self):
        icon = wx.Icon()
        icon.CopyFromBitmap(wx.Bitmap(icon_file, wx.BITMAP_TYPE_ANY))
        self.SetIcon(icon)

    def embed_browser(self, url):
        window_info = cef.WindowInfo()
        (width, height) = self.browser_panel.GetClientSize().Get()
        assert self.browser_panel.GetHandle(), "Window handle not available"
        window_info.SetAsChild(self.browser_panel.GetHandle(),
                               [0, 0, width, height])
        self.browser = cef.CreateBrowserSync(window_info,
                                             url=url)
        self.browser.SetClientHandler(FocusHandler())

    def OnSetFocus(self, _):
        if not self.browser:
            return
        if WINDOWS:
            cef.WindowUtils.OnSetFocus(self.browser_panel.GetHandle(),
                                       0, 0, 0)
        self.browser.SetFocus(True)

    def OnSize(self, _):
        if not self.browser:
            return
        if WINDOWS:
            cef.WindowUtils.OnSize(self.browser_panel.GetHandle(),
                                   0, 0, 0)
        elif LINUX:
            (x, y) = (0, 0)
            (width, height) = self.browser_panel.GetSize().Get()
            self.browser.SetBounds(x, y, width, height)
        self.browser.NotifyMoveOrResizeStarted()

    def OnClose(self, event):
        print("[wxpython.py] OnClose called")
        if not self.browser:
            # May already be closing, may be called multiple times on Mac
            return

        if MAC:
            # On Mac things work differently, other steps are required
            self.browser.CloseBrowser()
            self.clear_browser_references()
            self.Destroy()
            global g_count_windows
            g_count_windows -= 1
            if g_count_windows == 0:
                cef.Shutdown()
                wx.GetApp().ExitMainLoop()
                # Call _exit otherwise app exits with code 255 (Issue #162).
                # noinspection PyProtectedMember
                os._exit(0)
        else:
            # Calling browser.CloseBrowser() and/or self.Destroy()
            # in OnClose may cause app crash on some paltforms in
            # some use cases, details in Issue #107.
            self.browser.ParentWindowWillClose()
            event.Skip()
            self.clear_browser_references()

    def clear_browser_references(self):
        # Clear browser references that you keep anywhere in your
        # code. All references must be cleared for CEF to shutdown cleanly.
        self.browser = None


class FocusHandler(object):
    def OnGotFocus(self, browser, **_):
        # Temporary fix for focus issues on Linux (Issue #284).
        if LINUX:
            print("[wxpython.py] FocusHandler.OnGotFocus:"
                  " keyboard focus fix (Issue #284)")
            browser.SetFocus(True)


class CefApp(wx.App):

    def __init__(self, redirect):
        self.timer = None
        self.timer_id = 1
        self.is_initialized = False
        super(CefApp, self).__init__(redirect=redirect)

    def OnPreInit(self):
        super(CefApp, self).OnPreInit()
        # On Mac with wxPython 4.0 the OnInit() event never gets
        # called. Doing wx window creation in OnPreInit() seems to
        # resolve the problem (Issue #350).
        if MAC and wx.version().startswith("4."):
            print("[wxpython.py] OnPreInit: initialize here"
                  " (wxPython 4.0 fix)")
            self.initialize()

    def OnInit(self):
        self.initialize()
        return True

    def initialize(self):
        if self.is_initialized:
            return
        self.is_initialized = True
        self.create_timer()
        
        frame=wx.Frame(None)
        self.SetTopWindow(frame)
        TaskBarIcon(frame)
        pub.subscribe(self.on_receive_mg, "msg_update")
        
    def on_receive_mg(self, msg):
        browser_frame = MainFrame(msg)
        #self.SetTopWindow(browser_frame)
        browser_frame.Show()

    def create_timer(self):
        # See also "Making a render loop":
        # http://wiki.wxwidgets.org/Making_a_render_loop
        # Another way would be to use EVT_IDLE in MainFrame.
        self.timer = wx.Timer(self, self.timer_id)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(10)  # 10ms timer

    def on_timer(self, _):
        cef.MessageLoopWork()

    def OnExit(self):
        self.timer.Stop()
        return 0


if __name__ == '__main__':
    main()
