function spikeHound(varargin)
% Spike Hound - Scope v1
% This is the Scope and Real-Time Analysis Portion of the Spike Hound tool
% suite
%
% Initially Created Spring 2007 by Gus K. Lott III, PhD (guslott@yarcom.com)
%
% Copyright Gus Lott 2010
%
% This code is distributed under the GNU Public License
% http://www.gnu.org/licenses/gpl.txt
%




try
   
%Entry Switchyard
switch nargin
    case 0
        daqreset
        delete(findobj('tag','gSS07'))
        delete(findobj('tag','gSS07splash'))
        delete(findobj('tag','gSS07anal'))
        delete(findobj('tag','gFullScreen'))
        delete(findobj('tag','gSS07meta'))
        delete(timerfind)
        makegui
    case 1
        feval(varargin{1})
    case 2
        feval(varargin{1},varargin{2})
end

catch
    errordlg(lasterr)
    
    if ~isdeployed
        rethrow(lasterror)
    else 
        a=lasterror;
        a.message;
    end
end

% Construct initial Graphic Interface
function makegui
gui.version='1.2-OpenSource';

warning('OFF','daq:set:propertyChangeFlushedData')

%splash Screen
tempFig=figure('position',[0 0 714 220],'menubar','none','numbertitle','off','name',...
    ['Spike Hound v',gui.version],'tag','gSS07splash','resize','off');
centerfig(tempFig)
axes('position',[0 0 1 1],'color',[.8 .9 .8],'yticklabel',[]...
    ,'xticklabel',[],'xlim',[0 1],'ylim',[0 1],'xcolor',[.4 .4 .4],'ycolor',[.4 .4 .4],'visible','off');
box on
aTemp=imread('spikehoundSplash_Scope.jpg');
image(aTemp)
axis off

a=text(350,170,'Scanning for Interfaces','fontsize',12,'fontweight','bold',...
    'horizontalalignment','left');
pause(0.1); set(a,'string','Scanning for Interfaces...')

if isdeployed
%Register Interfaces on Stand-alone systems
interfaces={'keithley','mcc','nidaq','winsound'};
for i=1:length(interfaces)
    try 
        daqregister(interfaces{i});    
        set(a,'string',interfaces{i})
        pause(.15)
    end
end

ad=dir;
for i=1:length(ad)
    if ~isempty(strfind(ad(i).name,'.dll'))
        try
            daqregister(ad(i).name);    
            set(a,'string',ad(i).name)
            pause(.15)
        catch
            daqregister('parallel','unload')
        end
    end
end
end


warning off MATLAB:Axes:NegativeDataInLogAxis
%Initialize some Variables
gui.ax=[]; gui.pl=[]; gui.tx=[]; gui.ao=[]; gui.ai=[];
gui.aLink(1:4)=0; gui.ChanControls=[]; gui.OverFlag=0; gui.drop=0;
[gui.filt.b gui.filt.a]=butter(15,.25);


% Scan system for installed interfaces while splash screen is active
gTemp=daqhwinfo;

%Construct Main Figure
gui.fig=figure('name',['Spike Hound v',gui.version,' - Gus K. Lott III, PhD'],'numbertitle','off','tag','gSS07',...
    'menubar','none','position',[0 0 800 600],'resize','on','deletefcn',...
    @DelFcn,'doublebuffer','on');
try delete(tempFig); catch delete(gui.fig); return; end
centerfig(gui.fig)
set(gui.fig,'windowbuttonupfcn',@ChangeLevel)
set(gui.fig,'WindowScrollWheelFcn',@ScrollScopeScale)
set(gui.fig,'KeyPressFcn',@ScrollScopeScaleKey)


%Construct Axes for Display
gui.axtxt=uicontrol('style','text','units','normalized','position',...
    [0.2 0.96 0.3 0.03],'string','','fontunits','normalized','fontsize',1,...
    'backgroundcolor',get(gcf,'color'),'visible','off');
gui.backax=axes('position',[0.04 0.45 0.6 0.5],'xgrid','on','ygrid','on','color',...
    [.8 .9 .8],'yticklabel',[],'xticklabel',[],'XAxisLocation','top','ylim',[-1 1],...
    'box','on');
gui.timetext=text(0,1.15,'T','buttondownfcn',@DragTime,...
    'horizontalalignment','center','fontweight','bold');

gui.MainFigCapture=uicontrol('style','pushbutton','units','normalized',...
    'position',[0.62 0.925 0.02 0.025],'backgroundcolor',[.7 .2 .2],...
    'callback',{@FigureExtraction,gui.backax,'scope',[]},'userdata','figureextract');

gui.MainFigSave=uicontrol('style','pushbutton','units','normalized',...
    'position',[0.58 0.925 0.04 0.025],'backgroundcolor',[.2 .7 .2],'string','.MAT',...
    'foregroundcolor','w','callback',@CaptureScopeRaw);
gui.MainFigPause=uicontrol('style','toggle','units','normalized',...
    'position',[0.52 0.925 0.06 0.025],'backgroundcolor',[.7 .7 .7],'string','Pause',...
    'foregroundcolor','k');

gui.DataAnalysisMode=uicontrol('style','toggle','units','pixels',...
    'position',[537   571   256    24],'backgroundcolor',[1 1 .7],'string',...
    'Live Data Analysis','callback',@InitAnalysis,'fontweight','bold',...
    'fontunits','normalized','fontsize',.5,'tag','Data Analysis','enable','off');


%Multi-Option Selectors
gui.multiChan=uicontrol('style','Toggle','string','Chan',...
    'units','pixels','backgroundcolor',get(gcf,'color'),'position',...
    [5 241 48 18],'callback',@multiSelect,'value',1);
gui.multiBoard=uicontrol('style','Toggle','string','Input Cfg',...
    'units','pixels','backgroundcolor',get(gcf,'color'),'position',...
    [5   217    48    18],'callback',@multiSelect);
gui.multiNIDAQ=uicontrol('style','Toggle','string','NIDAQ',...
    'units','pixels','backgroundcolor',get(gcf,'color'),'position',...
    [5   193    48    18],'callback',@multiSelect);
gui.multiDIO=uicontrol('style','Toggle','string','Digital IO',...
    'units','pixels','backgroundcolor',get(gcf,'color'),'position',...
    [5 169 48 18],'callback',@multiSelect);
gui.multiStim=uicontrol('style','Toggle','string','Output',...
    'units','pixels','backgroundcolor',get(gcf,'color'),'position',...
    [5   145    48    18],'callback',@multiSelect);


%Frame for Channel Activation Controls
gui.ChannelAddPanel=uipanel('title','Available Channels','units','pixels',...
    'position',[57 145 224 120],'backgroundcolor',get(gcf,'color'),'visible','on');
gui.ChanDeviceName=uicontrol('parent',gui.ChannelAddPanel,'style','text','string',...
    'Select a Device',...
    'units','normalized','backgroundcolor',get(gcf,'color'),'position',...
    [0.05 0.75 0.9 0.17],'fontweight','bold');
gui.AvailableList=uicontrol('parent',gui.ChannelAddPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.2 0.5 0.4 0.2],'string',' ');
uicontrol('parent',gui.ChannelAddPanel,'style','text','string','Chan:','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.01 0.54 0.19 0.12],...
    'fontunits','normalized','fontsize',1);
gui.ChanColorSelect=uicontrol('parent',gui.ChannelAddPanel,'style','pushbutton',...
    'units','normalized','backgroundcolor','b','position',[0.62 0.5 0.2 0.2],...
    'string','Color','callback',@tracecolor,'foregroundcolor','w');
uicontrol('parent',gui.ChannelAddPanel,'style','text','string','Name:','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.01 0.34 0.19 0.10],...
    'fontunits','normalized','fontsize',1);
gui.ChanNameSet=uicontrol('parent',gui.ChannelAddPanel,'style','edit',...
    'backgroundcolor','w','units','normalized','position',[0.2 0.3 0.75 0.18],...
    'horizontalalignment','left');
gui.ChanAdd=uicontrol('parent',gui.ChannelAddPanel,'style','pushbutton','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.2 0.07 0.4 0.2],...
    'string','Add Channel','callback',@ChannelAdd);
gui.ChanDispQ=uicontrol('parent',gui.ChannelAddPanel,'style','Checkbox','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.65 0.07 0.32 0.2],...
    'string','Display','value',1);

%Frame for Board Config Details
gui.BoardConfigPanel=uipanel('title','Board Information & Configuration','units','pixels',...
    'position',[57 145 224 120],'backgroundcolor',get(gcf,'color'),'visible','off');
gui.BoardName=uicontrol('parent',gui.BoardConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',[.7 .7 1],'position',[0.01 0.85 0.98 0.15],...
    'string','','fontweight','bold');
gui.BoardDriver=uicontrol('parent',gui.BoardConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',[.7 .7 1],'position',[0.01 0.7 0.98 0.15],...
    'string','');
uicontrol('parent',gui.BoardConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.5 0.4 0.15],...
    'string','Input Type:','horizontalalignment','left');
gui.BoardInputType=uicontrol('parent',gui.BoardConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.5 0.53 0.45 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@BoardProperty);
uicontrol('parent',gui.BoardConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.3 0.4 0.15],...
    'string','In Clock Source:','horizontalalignment','left');
gui.BoardInClockSource=uicontrol('parent',gui.BoardConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.5 0.33 0.45 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@BoardProperty);
uicontrol('parent',gui.BoardConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.1 0.4 0.15],...
    'string','Channel Skew:','horizontalalignment','left');
gui.BoardSkewRate=uicontrol('parent',gui.BoardConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.5 0.13 0.45 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@BoardProperty);

%Frame for NIDAQ Specific Controls
gui.NIDAQConfigPanel=uipanel('title','NIDAQmx I/O Configuration','units','pixels',...
    'position',[57 145 224 120],'backgroundcolor',get(gcf,'color'),'visible','off');
uicontrol('parent',gui.NIDAQConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.02 0.8 0.65 0.15],...
    'string','InHwDigitalTriggerSource:','horizontalalignment','left');
gui.NIDAQHWTS=uicontrol('parent',gui.NIDAQConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.7 0.85 0.28 0.15],...
    'string',{' '},'horizontalalignment','left','callback',@BoardProperty,'enable','off');
uicontrol('parent',gui.NIDAQConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.02 0.6 0.65 0.15],...
    'string','InExternalSampleClockSource:','horizontalalignment','left');
gui.NIDAQESaCS=uicontrol('parent',gui.NIDAQConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.7 0.65 0.28 0.15],...
    'string',{' '},'horizontalalignment','left','callback',@BoardProperty,'enable','off');
uicontrol('parent',gui.NIDAQConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.02 0.4 0.65 0.15],...
    'string','InExternalScanClockSource:','horizontalalignment','left','callback',@BoardProperty);
gui.NIDAQEScCS=uicontrol('parent',gui.NIDAQConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.7 0.45 0.28 0.15],...
    'string',{' '},'horizontalalignment','left','callback',@BoardProperty,'enable','off');
uicontrol('parent',gui.NIDAQConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.02 0.2 0.65 0.15],...
    'string','OutExternalClockSource:','horizontalalignment','left');
gui.OutputECS=uicontrol('parent',gui.NIDAQConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.7 0.25 0.28 0.15],...
    'string',{' '},'horizontalalignment','left','callback',@BoardProperty,'enable','off');
uicontrol('parent',gui.NIDAQConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.02 0.00 0.65 0.15],...
    'string','OutHwDigitalTriggerSource:','horizontalalignment','left');
gui.OutputHWTS=uicontrol('parent',gui.NIDAQConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.7 0.05 0.28 0.15],...
    'string',{' '},'horizontalalignment','left','callback',@BoardProperty,'enable','off');

%Frame for Digital I/O Control
gui.DigitalConfigPanel=uipanel('title','Digital I/O','units','pixels',...
    'position',[57 145 224 120],'backgroundcolor',get(gcf,'color'),'visible','off');
gui.DIOSelect=uicontrol('parent',gui.DigitalConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.05 0.83 0.55 0.15],...
    'string',{' '},'horizontalalignment','left');
gui.DIOConnect=uicontrol('parent',gui.DigitalConfigPanel,'style','Toggle','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.65 0.8 0.32 0.18],...
    'string','Connect','horizontalalignment','left','callback',@digitalConnect);
gui.DIOLines=uicontrol('parent',gui.DigitalConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.05 0.58 0.55 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@selectDIOport);
gui.DIOaddIn=uicontrol('parent',gui.DigitalConfigPanel,'style','pushbutton','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.65 0.55 0.15 0.18],...
    'string','In','horizontalalignment','left','enable','off','callback',@addDIOport);
gui.DIOaddOut=uicontrol('parent',gui.DigitalConfigPanel,'style','pushbutton','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.82 0.55 0.15 0.18],...
    'string','Out','horizontalalignment','left','enable','off','callback',@addDIOport);
uicontrol('parent',gui.DigitalConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.3 0.25 0.15],...
    'string','Rec Rate:','horizontalalignment','left');
gui.DIORate=uicontrol('parent',gui.DigitalConfigPanel,'style','edit','units',...
    'normalized','backgroundcolor','w','position',[0.3 0.3 0.2 0.18],...
    'string','50','userdata',50,'enable','off','callback',...
    ['if isnan(str2double(get(gcbo,''string''))); set(gcbo,''string'',num2str(get(gcbo,''userdata'')));',...
    'else set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end']);
gui.DIOFile=uicontrol('parent',gui.DigitalConfigPanel,'style','pushbutton','tag','temp.txt','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.55 0.3 0.42 0.18],...
    'string','temp.txt','horizontalalignment','left','enable','off','callback',@DIOLogFile);
gui.DIOLogChange=uicontrol('parent',gui.DigitalConfigPanel,'style','checkbox','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.42 0.15 0.55 0.14],...
    'string','Log Changes Only','horizontalalignment','left','enable','off');
gui.DIOLogLink=uicontrol('parent',gui.DigitalConfigPanel,'style','checkbox','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.42 0.01 0.55 0.14],...
    'string','Link with Recording','horizontalalignment','left','enable','off');
gui.DIOLog=uicontrol('parent',gui.DigitalConfigPanel,'style','Toggle','units',...
    'normalized','backgroundcolor',[1 .7 .7],'position',[0.05 0.05 0.35 0.22],...
    'string','Start Logging','horizontalalignment','left','enable','off','callback',@DIOLogRun);

%Frame for Stimulus control
gui.OutputConfigPanel=uipanel('title','Signal Generation','units','pixels',...
    'position',[57 145 224 120],'backgroundcolor',get(gcf,'color'),'visible','off');
gui.OutputState(1)=uicontrol('parent',gui.OutputConfigPanel,'style','Toggle','units',...
    'normalized','backgroundcolor',[1 1 .7],'position',[0.05 0.82 0.9 0.18],...
    'string','Pulse Train Generation','horizontalalignment','left','value',1,'callback',@StimSwitch,'enable','off');
gui.OutputState(2)=uicontrol('parent',gui.OutputConfigPanel,'style','Toggle','units',...
    'normalized','backgroundcolor',[1 .7 1],'position',[0.05 0.63 0.9 0.18],...
    'string','Function Generator','horizontalalignment','left','callback',@StimSwitch,'enable','off');

uicontrol('parent',gui.OutputConfigPanel,'style','text','backgroundcolor',...
    get(gcf,'color'),'string','Sample Rate (Hz)','units',...
    'normalized','position',[0.05 0.43 0.65 0.15],'horizontalalignment','left');
gui.StimSRate=uicontrol('parent',gui.OutputConfigPanel,'style','edit','string','10000',...
    'userdata',10000,'units','normalized','position',[0.5 0.45 0.4 0.17],...
    'backgroundcolor','w','tag','gui.StimSRate','enable','off');

uicontrol('parent',gui.OutputConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.23 0.65 0.15],...
    'string','Output Range:','horizontalalignment','left');
gui.OutputRange=uicontrol('parent',gui.OutputConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.5 0.28 0.4 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@BoardProperty);

uicontrol('parent',gui.OutputConfigPanel,'style','text','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.05 0.05 0.65 0.15],...
    'string','ClockSource:','horizontalalignment','left');
gui.OutputCS=uicontrol('parent',gui.OutputConfigPanel,'style','popupmenu','units',...
    'normalized','backgroundcolor','w','position',[0.5 0.08 0.4 0.15],...
    'string',{' '},'horizontalalignment','left','enable','off','callback',@BoardProperty);


%Frame for Channel Select Controls
gui.ChannelListPanel=uipanel('title','Active Channels','units','pixels',...
    'position',[289 145 224 120],'backgroundcolor',get(gcf,'color'));
gui.ChanList=uicontrol('parent',gui.ChannelListPanel,'style','listbox','units',...
    'normalized','position',[0.05 0.3 0.9 0.65],'backgroundcolor','w','callback',...
    @DisplayChanControls);
gui.RemoveChan=uicontrol('parent',gui.ChannelListPanel,'style','pushbutton','units',...
    'normalized','position',[0.05 0.05 0.6 0.2],'string','Remove Channel',...
    'backgroundcolor',get(gcf,'color'),'callback',@ChannelRemove);
gui.ChanAudio=uicontrol('parent',gui.ChannelListPanel,'style','Checkbox','units',...
    'normalized','backgroundcolor',get(gcf,'color'),'position',[0.67 0.05 0.32 0.2],...
    'string','Listen','value',0,'callback',@AudioFeedback);

%Frame for Recording Controls
gui.RecControls=uipanel('title','Record','units','pixels','position',...
    [521   427   272   138],'backgroundcolor',get(gcf,'color'));
gui.RecBrowse=uicontrol('parent',gui.RecControls,'style','pushbutton','units',...
    'normalized','position',[0.02 0.76 0.3 0.2],'backgroundcolor',get(gcf,'color'),...
    'string','File Name...','callback',@recbrowse);
gui.RecDir=uicontrol('parent',gui.RecControls,'style','pushbutton','units',...
    'normalized','position',[0.37 0.76 0.3 0.2],'backgroundcolor',get(gcf,'color'),...
    'string','Open Dir','tag',pwd,'callback',@openpwd);
gui.RecMeta=uicontrol('parent',gui.RecControls,'style','toggle','units',...
    'normalized','position',[0.72 0.76 0.26 0.2],'backgroundcolor',get(gcf,'color'),...
    'string','Metadata >','tag','','callback',@OpenMeta);
gui.RecFile=uicontrol('parent',gui.RecControls,'style','text','units','normalized',...
    'position',[0.02 0.48 0.96 0.26],'backgroundcolor',[0.6 0.6 0.6],...
    'string',[pwd,'\temp.daq'],'horizontalalignment','left');
gui.Rec(1)=uicontrol('parent',gui.RecControls,'style','radio','units','normalized',...
    'position',[0.02 0.25 0.96 0.2],'backgroundcolor',get(gcf,'color'),...
    'string','Fixed Duration (s)    >>>>>>');
    gui.RecDuration=uicontrol('parent',gui.RecControls,'style','text','units',...
        'normalized','position',[0.6 0.25 0.38 0.2],'backgroundcolor',...
        get(gui.RecControls,'backgroundcolor'),'foregroundcolor',...
    get(gui.RecControls,'backgroundcolor'),'string','10','userdata',1,'tag','gRecDur',...
        'callback',['if isnan(str2double(get(gcbo,''string'')));',...
        'set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else;',...
        'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end;'],...
        'userdata',0);
gui.Rec(2)=uicontrol('parent',gui.RecControls,'style','radio','units','normalized',...
    'position',[0.02 0.05 0.4 0.2],'backgroundcolor',get(gcf,'color'),...
    'string','Until Stopped','value',1);
set(gui.Rec(2),'userdata',gui.Rec,'callback',...
    ['set(get(gcbo,''userdata''),''value'',0); set(gcbo,''value'',1);',...
    ' set(findobj(''tag'',''gRecDur''),''style'',''text'',''backgroundcolor'',',...
    'get(get(gcbo,''parent''),''backgroundcolor''),''foregroundcolor'','...
    'get(get(gcbo,''parent''),''backgroundcolor''));'])
set(gui.Rec(1),'userdata',gui.Rec,'callback',...
    ['set(get(gcbo,''userdata''),''value'',0); set(gcbo,''value'',1);',...
    ' set(findobj(''tag'',''gRecDur''),''style'',''edit'',''backgroundcolor'',',...
    '''w'',''foregroundcolor'',''k'');'])
gui.RecStartStop=uicontrol('parent',gui.RecControls,'style','toggle','units',...
    'normalized','position',[0.5 0.02 0.48 0.2],'backgroundcolor','r','string',...
    'Record Start','callback',@GoStartStop);

%Frame for Trigger Controls
gui.TrigControls=uipanel('title','Trigger','units','pixels','position',...
    [521   271   136   150],'backgroundcolor',get(gcf,'color'),'foregroundcolor','r');
gui.Trig(1)=uicontrol('parent',gui.TrigControls,'style','radio','units','normalized',...
    'position',[0.02 0.85 .96 0.15],'string','Continuous','backgroundcolor',...
    get(gui.TrigControls,'backgroundcolor'),'value',1,'callback',@ChangeTrig);
gui.Trig(2)=uicontrol('parent',gui.TrigControls,'style','radio','units','normalized',...
    'position',[0.02 0.7 .96 0.15],'string','Manual','backgroundcolor',...
    get(gui.TrigControls,'backgroundcolor'),'callback',@ChangeTrig);
    gui.TrigMan=uicontrol('parent',gui.TrigControls,'style','Pushbutton','units',...
        'normalized','position',[0.55 0.7 0.43 0.15],'string','Trigger',...
        'backgroundcolor',get(gui.TrigControls,'backgroundcolor'),'callback',...
        'trigger(get(gcbo,''userdata''))','enable','off');
gui.Trig(3)=uicontrol('parent',gui.TrigControls,'style','radio','units','normalized',...
    'position',[0.02 0.55 .96 0.15],'string','Channel','backgroundcolor',...
    get(gui.TrigControls,'backgroundcolor'));

    uicontrol('parent',gui.TrigControls,'style','text','units','normalized',...
        'position',[0.2 0.43 0.3 0.10],'string','Chan:','backgroundcolor',...
        get(gcf,'color'),'horizontalalignment','left');
    gui.TrigChan=uicontrol('parent',gui.TrigControls,'style','popupmenu','units',...
        'normalized','position',[0.5 0.4 0.48 0.15],'string',' ','backgroundcolor',...
        'w','enable','off');
    uicontrol('parent',gui.TrigControls,'style','text','units','normalized',...
        'position',[0.2 0.25 0.4 0.10],'string','Level:','backgroundcolor',...
        get(gcf,'color'),'horizontalalignment','left');
    gui.TrigLevel=uicontrol('parent',gui.TrigControls,'style','edit','units',...
        'normalized','position',[0.5 0.23 0.48 0.15],'string','0.05 ',...
        'backgroundcolor','w','enable','off','userdata',0.05);
    
gui.Trig(4)=uicontrol('parent',gui.TrigControls,'style','radio','units','normalized',...
    'position',[0.02 0.11 .96 0.12],'string','External (HWDigital)',...
    'backgroundcolor',get(gui.TrigControls,'backgroundcolor'));
gui.Trig(5)=uicontrol('parent',gui.TrigControls,'style','radio','units','normalized',...
    'position',[0.02 0.01 .96 0.12],'string','Stimulus (Software)',...
    'backgroundcolor',get(gui.TrigControls,'backgroundcolor'));
set(gui.Trig,'userdata',gui.Trig,'callback',{@ChangeTrig,'TrigRadio'})

%Frame for Time Controls
gui.TimeControls=uipanel('title','Time','units','pixels','position',...
    [689   271   104   150],'backgroundcolor',get(gcf,'color'),'foregroundcolor','r');
uicontrol('parent',gui.TimeControls,'style','text','string','Sample Rate (Hz)',...
    'backgroundcolor',get(gcf,'color'),'units','normalized','position',...
    [0.05 0.8 .9 0.1],'horizontalalignment','left');
gui.SRate=uicontrol('parent',gui.TimeControls,'style','Edit','string','10000',...
    'backgroundcolor','w','units','normalized','position',[0.05 0.6 .9 0.2],...
    'userdata',10000,'callback',@initAI);
uicontrol('parent',gui.TimeControls,'style','text','string','Full Span (s)',...
    'backgroundcolor',get(gcf,'color'),'units','normalized','position',...
    [0.05 0.48 .9 0.1],'horizontalalignment','left');
gui.Refresh=uicontrol('parent',gui.TimeControls,'style','popupmenu','string',...
    {'0.05','0.1','0.2','0.4','0.6','0.8','1','2','5','10','20','30'},...
    'backgroundcolor','w','units','normalized','position',[0.05 0.28 .9 0.2],...
    'userdata',0.1,'value',2,'callback',@initAI);
uicontrol('parent',gui.TimeControls,'style','text','string','Time Offset (%)',...
    'backgroundcolor',get(gcf,'color'),'units','normalized','position',...
    [0.05 0.18 .9 0.1],'horizontalalignment','left');
gui.TriggerDelay=uicontrol('parent',gui.TimeControls,'style','edit','string','0',...
    'backgroundcolor','w','units','normalized','position',[0.05 0.01 .9 0.17],...
    'userdata',0,'callback',@ChangeTrig);

gui.FunctionGenerator=uipanel('title','Function Generator','units','pixels','position',...
    [5 7 788 132],'backgroundcolor',[1 .7 1],'visible','off');
%Sin,Square,Triangle,Noise, DC
%Frequency, Duty Cycle, Phase, Amplitude
%Add to current Signal
%Reset Current Signal
%Duration or Indefinite
%Use AO timerfcn to pack new samples into output to prevent phase clicks when generating
gui.FCNtype=uicontrol('parent',gui.FunctionGenerator,'style','popupmenu',...
    'string',{'Sine','Square','Triangle','White Noise','DC Offset'},'units','normalized','position',[0.01 0.95 0.1 0.05],...
    'backgroundcolor','w');
gui.FCNtypeAdd=uicontrol('parent',gui.FunctionGenerator,'style','pushbutton',...
    'string','Add','units','normalized','position',[0.12 0.83 0.1 0.15],'callback',@FCNAddElement);
gui.FCNtypeList=uicontrol('parent',gui.FunctionGenerator,'style','listbox',...
    'string',{},'value',0,'units','normalized','position',[0.01 0.2 0.21 0.6],'backgroundcolor','w','callback',@FCNSelectElement);
gui.FCNtypeRemove=uicontrol('parent',gui.FunctionGenerator,'style','pushbutton',...
    'string','Remove','units','normalized','position',[0.01 0.03 0.1 0.15],'callback',@FCNRemoveElement);
gui.FCNtypeClear=uicontrol('parent',gui.FunctionGenerator,'style','pushbutton',...
    'string','Clear All','units','normalized','position',[0.12 0.03 0.1 0.15],'callback',@FCNClearAll);

gui.FCNOutAx=axes('parent',gui.FunctionGenerator,'position',[0.6 0.18 0.38 0.77],'xaxislocation','top',...
    'fontsize',6,'xticklabel',[],'color',[1 .9 1],'box','on','ygrid','on','xgrid','on');
gui.FCNOutPl=line(nan,nan,'parent',gui.FCNOutAx,'userdata',[],'linewidth',2);
gui.FCNTitle=title('');
gui.FCNtypeGen=uicontrol('parent',gui.FunctionGenerator,'style','toggle',...
    'string','Generate Signal','units','normalized','position',[0.6 0.01 0.38 0.15],'callback',@FCNGenGo);

%Frame For Stimulation Control ---------------------------------
gui.StimControls=uipanel('title','Pulse Train Stimulus Generation','units','pixels','position',...
    [5 7 788 132],'backgroundcolor',[1 1 .7]);

%Mode Controls (Single/Continuous)
gui.StimMode(1)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.01 0.7 0.11 0.2],'string','Single','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'value',1);
gui.StimModeTrig=uicontrol('parent',gui.StimControls,'style','pushbutton','units',...
    'normalized','position',[0.12 0.72 0.08 0.18],'string','Trigger',...
    'backgroundcolor',get(gui.fig,'color'),'tag','gSS07ao',...
    'callback',@StimLoad);
gui.StimMode(2)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.01 0.5 0.11 0.2],'string','Continuous',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'));
gui.StimModeStart=uicontrol('parent',gui.StimControls,'style','toggle','units',...
    'normalized','position',[0.12 0.52 0.08 0.18],'string','Start','backgroundcolor',...
    get(gui.fig,'color'),'enable','off','callback',['spikeHound(''StimLoad'');',...
    'if get(gcbo,''value'')==1; start(get(gcbo,''userdata'')); ',...
    'set(gcbo,''string'',''Stop''); else; stop(get(gcbo,''userdata'')); ',...
    'set(gcbo,''string'',''Start''); end']);
gui.StimModeHWCheck=uicontrol('parent',gui.StimControls,'style','checkbox','units',...
    'normalized','position',[0.01 0.3 0.2 0.2],'string','HwDigital Trigger',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'),'callback','spikeHound(''StimLoad'')');
set(gui.StimMode,'userdata',gui.StimMode,'callback',...
    ['set(get(gcbo,''userdata''),''value'',0); ',...
    'set(gcbo,''value'',1); spikeHound(''StimParam'')'])

%Waveform Shape (pulse duration,3x delays)
gui.StimType(1)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.22 0.8 0.10 0.15],'string','One Pulse',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'),'value',1);
gui.StimType(2)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.22 0.65 0.10 0.15],'string','Two Pulses',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'));
gui.StimType(3)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.22 0.5 0.10 0.15],'string','Tetanic',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'));
gui.StimType(4)=uicontrol('parent',gui.StimControls,'style','radio','units',...
    'normalized','position',[0.22 0.34 0.10 0.15],'string','',...
    'backgroundcolor',get(gui.StimControls,'backgroundcolor'));
gui.StimTypeLoad=uicontrol('parent',gui.StimControls,'style','PushButton','units',...
    'normalized','position',[0.24 0.35 0.08 0.15],'string','Load Script',...
    'enable','off','callback',@LoadAStim);
set(gui.StimType,'userdata',gui.StimType,'callback',...
    ['set(get(gcbo,''userdata''),''value'',0);',...
    ' set(gcbo,''value'',1); spikeHound(''StimParam'')'])

uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
get(gui.StimControls,'backgroundcolor'),'string','Pulse Dur','units','normalized',...
'position',[0.45 0.83 0.08 0.14],'horizontalalignment','left');
gui.StimShape(1)=uicontrol('parent',gui.StimControls,'style','edit','string','0.005',...
    'userdata',0.005,'units','normalized','position',[0.53 0.83 0.1 0.14],...
    'backgroundcolor',[1 .7 .7],'tag','PulseDur');
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Delay 1','units','normalized',...
    'position',[0.45 0.68 0.1 0.14],'horizontalalignment','left');
gui.StimShape(2)=uicontrol('parent',gui.StimControls,'style','edit','string','0.005',...
    'userdata',0.005,'units','normalized','position',[0.53 0.68 0.1 0.14],...
    'backgroundcolor',[.7 1 .7],'tag','Delay1');
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Delay 2','units','normalized',...
    'position',[0.45 0.53 0.1 0.14],'horizontalalignment','left');
gui.StimShape(3)=uicontrol('parent',gui.StimControls,'style','edit','string','0.010',...
    'userdata',0.010,'units','normalized','position',[0.53 0.53 0.1 0.14],...
    'backgroundcolor',[.9 .7 1],'tag','Delay2');
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Delay 3','units','normalized',...
    'position',[0.45 0.38 0.1 0.14],'horizontalalignment','left');
gui.StimShape(4)=uicontrol('parent',gui.StimControls,'style','edit','string','0.010',...
    'userdata',0.010,'units','normalized','position',[0.53 0.38 0.1 0.14],...
    'backgroundcolor',[.7 1 1],'tag','Delay3');
set(gui.StimShape,'callback',...
    ['if isnan(str2double(get(gcbo,''string'')));',...
    ' set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else; ',...
    'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end; ',...
    'spikeHound(''StimParam'')'])

%Tetanus duration, Tetanic Interval
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Tet. Duration:','units',...
    'normalized','position',[0.65 0.83 0.15 0.14],'horizontalalignment','left');
gui.StimTetDur=uicontrol('parent',gui.StimControls,'style','edit','string','0.1',...
    'userdata',0.1,'units','normalized','position',[0.75 0.83 0.1 0.14],...
    'backgroundcolor',[.9 .9 .6],'tag','TetDur');
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Tet. Interval:','units',...
    'normalized','position',[0.65 0.68 0.15 0.14],'horizontalalignment','left');
gui.StimTetInt=uicontrol('parent',gui.StimControls,'style','edit','string','0.01',...
    'userdata',0.01,'units','normalized','position',[0.75 0.68 0.1 0.14],...
    'backgroundcolor',[1 .9 .4],'tag','TetInt');
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Repeat (s):','units',...
    'normalized','position',[0.65 0.5 0.15 0.14],'horizontalalignment','left');
%repeat interval and pulse amplitudes
gui.StimRepeat=uicontrol('parent',gui.StimControls,'style','edit','string','1',...
    'userdata',1,'units','normalized','position',[.75 .5 .1 .14],'backgroundcolor','w');
set([gui.StimTetDur gui.StimTetInt],'callback',...
    ['if isnan(str2double(get(gcbo,''string'')));',...
    ' set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else; ',...
    'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end; ',...
    'spikeHound(''StimParam'')'])
uicontrol('parent',gui.StimControls,'style','text','backgroundcolor',...
    get(gui.StimControls,'backgroundcolor'),'string','Amplitude (V)','units',...
    'normalized','position',[0.65 0.35 0.15 0.14],'horizontalalignment','left');
gui.StimAmp=uicontrol('parent',gui.StimControls,'style','edit','string','10',...
    'userdata',10,'units','normalized','position',[.75 .35 .1 .14],...
    'backgroundcolor','w','tag','gui.StimAmp');

set([gui.StimSRate gui.StimAmp gui.StimRepeat],'callback',...
    ['if isnan(str2double(get(gcbo,''string''))); ',...
    'set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else; ',...
    'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end;',...
    ' spikeHound(''StimParam'')'])

%Graph for waveform
AxLoc=[0.005 0.005 .99 .3];
gui.StimBackAx=axes('parent',gui.StimControls,'color',[.4 .4 .4],'position',AxLoc,...
    'xticklabel',[],'ytick',[]);
box on
gui.StimAx=axes('parent',gui.StimControls,'color','none','position',AxLoc,'xtick',[],...
    'ytick',[],'ylim',[0 11],'xlim',[0 .1]);
gui.StimPl=line(0,5,'color','w','linewidth',2);
gui.StimPlPulseDur=line([0 0],[8.5 8.5],'color',get(gui.StimShape(1),...
    'backgroundcolor'),'linewidth',3,'userdata',8.5/10);
gui.StimPlDelay1=line([0 0],[5.5 5.5],'color',get(gui.StimShape(2),...
    'backgroundcolor'),'linewidth',3,'userdata',5.5/10);
gui.StimPlDelay2=line([0 0],[5.5 5.5],'color',get(gui.StimShape(3),...
    'backgroundcolor'),'linewidth',3,'userdata',5.5/10);
gui.StimPlDelay3=line([0 0],[5.5 5.5],'color',get(gui.StimShape(4),...
    'backgroundcolor'),'linewidth',3,'userdata',5.5/10);
gui.StimPlTetDur=line([0 0],[2.5 2.5],'color',get(gui.StimTetDur,...
    'backgroundcolor'),'linewidth',3,'userdata',2.5/10);
gui.StimPlTetInt=line([0 0],[8.5 8.5],'color',get(gui.StimTetInt,...
    'backgroundcolor'),'linewidth',3,'userdata',8.5/10);

gui.StimViolation=uicontrol('parent',gui.StimControls,'style','text',...
    'backgroundcolor',[.9 .9 .6],'units','normalized','position',AxLoc,...
   'foregroundcolor','r','string','Stimulus Parameter Error','fontunits','normalized',...
    'fontsize',.8,'fontweight','bold','visible','off');

gui.AIControls=[gui.ChannelAddPanel gui.ChannelListPanel gui.RecControls...
    gui.TrigControls gui.TimeControls];

for i=[gui.AIControls,   gui.StimControls]
    a=get(i,'children');
    b=findobj(a,'type','axes');
    for j=1:length(b)
        a(a==b(j))=[];
    end
    set(a,'enable','off')
end


%--------------------------------------------
%Construct menus for interface selection based on available hardware
gui.HWMenu=uimenu('Label','Input','userdata',0);
gui.OutMenu=uimenu('Label','Output','userdata',0);
%Construct other Dropdown menus

gui.DataMenu(1)=uimenu('Label','File Tools');
gui.DataMenu(end+1)=uimenu('parent',gui.DataMenu(1),'Label','Display .daq File',...
    'callback','spikeHound(''LoadDaq'')','separator','on');
gui.DataMenu(end+1)=uimenu('parent',gui.DataMenu(1),'Label',...
    'Open .fig File','callback','spikeHound(''OpenFIG'')');
gui.DataMenu(end+1)=uimenu('parent',gui.DataMenu(1),'Label','Convert .daq to .mat',...
    'callback','spikeHound(''ConvertMAT'')','separator','on');
gui.DataMenu(end+1)=uimenu('parent',gui.DataMenu(1),'Label','Convert .daq to .txt',...
    'callback','spikeHound(''ConvertTXT'')');
gui.DataMenu(end+1)=uimenu('parent',gui.DataMenu(1),'Label',...
    'Convert .daq to .wav audio','callback','spikeHound(''ConvertWAV'')');

gui.AdvancedMenu(1)=uimenu('Label','Advanced');
gui.AdvancedMenu(3)=uimenu('parent',gui.AdvancedMenu(1),'Label',...
    'Save Scope Connection State','Separator','on','callback',@SaveGPRIMEState);
gui.AdvancedMenu(4)=uimenu('parent',gui.AdvancedMenu(1),'Label',...
    'Load Scope Connection State','callback',@LoadGPRIMEState);
gui.AdvancedMenu(5)=uimenu('parent',gui.AdvancedMenu(1),'Label','Measure off Active Trace',...
    'callback','spikeHound(''MeasureGo'')','separator','on');
gui.AdvancedMenu(6)=uimenu('parent',gui.AdvancedMenu(1),'Label','Full Screen Scope Mode',...
    'callback',@FullScreenScope);
gui.AdvancedMenu(7)=uimenu('parent',gui.AdvancedMenu(1),'Label','Streaming Scope Display','separator','on','checked','on',...
    'callback','if strcmpi(get(gcbo,''checked''),''on''); set(gcbo,''checked'',''off''); else set(gcbo,''checked'',''on''); end');
% gui.AdvancedMenu(2)=uimenu('parent',gui.AdvancedMenu(1),'separator','on','Label',...
%     'NI DAQmx Signal Routing','callback','LottGraphicalDAQmxRoute');
% gui.AdvancedMenu(8)=uimenu('parent',gui.AdvancedMenu(1),'Label',...
%     'NI DAQmx Clocked Digital Out','enable','off','callback','');
gui.AdvancedMenu(9)=uimenu('parent',gui.AdvancedMenu(1),'Label',...
    'Graphics Acceleration','enable','on','separator','on');
s=set(gui.fig,'WVisual');
gTemp2=get(gui.fig,'WVisual');
for i=1:length(s)
    set(gui.fig,'WVisual',s{i})
    gT=uimenu('parent',gui.AdvancedMenu(9),'Label',get(gui.fig,'WVisual'),'callback',@RenderChange);
    if strcmpi(get(gui.fig,'WVisual'),gTemp2); set(gT,'checked','on'); end
end
set(gui.fig,'WVisual',gTemp2)

gui.AboutMenu(1)=uimenu('Label','About');
gui.AboutMenu(end+1)=uimenu('parent',gui.AboutMenu(1),'Label',...
    'Launch g-PRIME Web Site (freeware)','callback',@launchWeb);
gui.AboutMenu(end+1)=uimenu('parent',gui.AboutMenu(1),'Label',...
    'GK Lott''s Dissertation','callback',@launchRef);
gui.AboutMenu(end+1)=uimenu('parent',gui.AboutMenu(1),'Label',...
    'Spike Hound User''s Manual','callback',@LaunchManual,'separator','on');
gui.AboutMenu(end+1)=uimenu('parent',gui.AboutMenu(1),'Label',...
    'About Spike Hound','callback',@Aboutgprime,'separator','off');


j=1; p=1;
dIOs={};
dIO={};
for i=1:length(gTemp.InstalledAdaptors)
try gFoo=daqhwinfo(gTemp.InstalledAdaptors{i}); catch continue; end
try
    if ~isempty(gFoo.ObjectConstructorName{1,1})
        gui.HWSubMenu(j)=uimenu('parent',gui.HWMenu,'Label',gTemp.InstalledAdaptors{i});
        for k=1:length(gFoo.InstalledBoardIds)
            if ~isempty(gFoo.ObjectConstructorName{k,1})
                uimenu('parent',gui.HWSubMenu(j),'Label',gFoo.BoardNames{k},...
                    'callback',{@BoardSelect,gTemp.InstalledAdaptors{i},gFoo.InstalledBoardIds{k},k},'tag','gInMenu');           
            end
        end
        j=j+1;
    end
end
try
    if ~isempty(gFoo.ObjectConstructorName{1,2})
        gui.OutSubMenu(p)=uimenu('parent',gui.OutMenu,'Label',gTemp.InstalledAdaptors{i});
        for k=1:length(gFoo.InstalledBoardIds)
            if ~isempty(gFoo.ObjectConstructorName{k,2})
                uimenu('parent',gui.OutSubMenu(p),'Label',gFoo.BoardNames{k},...
                    'callback',{@OutputBoardSelect,gTemp.InstalledAdaptors{i},gFoo.InstalledBoardIds{k},k},'tag','gOutMenu');           
            end
        end
        p=p+1;
    end
end
try
    for k=1:length(gFoo.InstalledBoardIds)
        if ~isempty(gFoo.ObjectConstructorName{k,3})
            dIOs{end+1}=[gFoo.BoardNames{k},'/',gFoo.InstalledBoardIds{k}];
            dIO{end+1,1}=gTemp.InstalledAdaptors{i};
            dIO{end,2}=gFoo.InstalledBoardIds{k};
        end
    end
end
end
if isempty(dIOs); dIOs={' '}; end
set(gui.DIOSelect,'string',dIOs,'userdata',dIO);

set(gui.fig,'userdata',gui)
set(gui.DataMenu(1),'userdata',[pwd,'\'])
set(gui.RecDir,'enable','on')

ad=dir;
for i=1:length(ad)
    if strcmpi(ad(i).name,'default.mat')
        LoadGPRIMEState([],'default');
    end
end
FigureExtraction([],[],[],'Update',[])
set(gui.fig,'resizefcn',@figResize)

function ScrollScopeScale(obj,event)
%Scale Data Display Axes with the mouse wheel based on divisions in the
%channel VPD box
try
gui=get(findobj('tag','gSS07'),'userdata');
Chan=get(findobj(gui.ChanControls,'visible','on','type','uipanel'),'userdata');
currentVal=get(Chan.VpD,'value');
maxVal=length(get(Chan.VpD,'string'));
Val=currentVal-event.VerticalScrollCount;
if Val>maxVal; Val=maxVal; end
if Val<1; Val=1; end
set(Chan.VpD,'value',Val)
scaleAx
end

%Scroll w/ +/- keys on keyboard
function ScrollScopeScaleKey(obj,event)
if ~(strcmpi(event.Character,'+')|strcmpi(event.Character,'-'))
    return
end
try
    gui=get(findobj('tag','gSS07'),'userdata');
    Chan=get(findobj(gui.ChanControls,'visible','on','type','uipanel'),'userdata');
    currentVal=get(Chan.VpD,'value');
    maxVal=length(get(Chan.VpD,'string'));    
    switch event.Character
        case '+'
            Val=currentVal+1;
        case '-'
            Val=currentVal-1;
    end
    if Val>maxVal; Val=maxVal; end
    if Val<1; Val=1; end
    set(Chan.VpD,'value',Val)
    scaleAx
end

function RenderChange(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
figs=findobj('type','figure');
set(figs,'WVisual',get(obj,'Label'))
if strfind(get(obj,'label'),'Hardware Accelerated')
    set(figs,'renderer','OpenGL')
elseif strfind(get(obj,'label'),'Opengl')
    set(figs,'renderer','OpenGL')
else
    set(figs,'renderermode','auto')
end

set(get(get(obj,'parent'),'children'),'checked','off');
set(obj,'checked','on')

%Resize the figure, scale the axis only
function figResize(obj,event)
gui=get(obj,'userdata');
%Don't go below 800x600
screensize=get(0,'ScreenSize');

pos=get(obj,'position');
if pos(3)<800; pos(3)=800; end
if pos(4)<600; pos(4)=600; end

if screensize(3)<(pos(1)+pos(3))
    pos(1)=screensize(3)-pos(3);
end
if screensize(4)<(pos(2)+pos(4)+40)
    pos(2)=screensize(4)-pos(4)-40;
end

set(obj,'position',pos);
right=pos(3);
top=pos(4);


%if fig is big enough to drop record controls down to bottom, do that
gT=get(gui.StimControls,'position');
if (right-gT(1)-gT(3))>280
    %Drop Controls to bottom
    gTemp=get(gui.RecControls,'position'); gTemp(1)=797; gTemp(2)=7+156; set(gui.RecControls,'position',gTemp);
    gTemp=get(gui.TrigControls,'position'); gTemp(1)=797; gTemp(2)=7; set(gui.TrigControls,'position',gTemp);
    gTemp=get(gui.TimeControls,'position'); gTemp(1)=797+168; gTemp(2)=7; set(gui.TimeControls,'position',gTemp);
    gTemp=get(gui.DataAnalysisMode,'position'); gTemp(1)=521; gTemp(2)=300-24; set(gui.DataAnalysisMode,'position',gTemp);
    gui.drop=1;
else
    %Place Controls at right
    gTemp=get(gui.RecControls,'position'); gTemp(1)=right-(800-521); gTemp(2)=427; set(gui.RecControls,'position',gTemp);
    gTemp=get(gui.TrigControls,'position'); gTemp(1)=right-(800-521); gTemp(2)=271; set(gui.TrigControls,'position',gTemp);
    gTemp=get(gui.TimeControls,'position'); gTemp(1)=right-(800-689); gTemp(2)=271; set(gui.TimeControls,'position',gTemp);
    gTemp=get(gui.DataAnalysisMode,'position'); gTemp(1)=right-(800-537); gTemp(2)=571; set(gui.DataAnalysisMode,'position',gTemp);
    gui.drop=0;
end

%Scale Axis
if get(gui.backax,'parent')==gui.fig
    if gui.drop==1
        axpos(1)=0.04;
        gTemp=get(gui.DataAnalysisMode,'position');
        axpos(2)=(gTemp(2)+gTemp(4))/pos(4)+0.02+0.04*get(gui.DIOConnect,'value');
        axpos(3)=0.92;
        axpos(4)=0.95-axpos(2);
        set([gui.backax,gui.ax],'position',axpos)
    else
        axpos(1)=0.04;
        gTemp=get(gui.ChannelAddPanel,'position');
        axpos(2)=(gTemp(2)+gTemp(4))/pos(4)+0.02+0.04*get(gui.DIOConnect,'value');
        gTemp=get(gui.RecControls,'position');
        axpos(3)=0.95-(pos(3)-gTemp(1))/pos(3);
        axpos(4)=0.95-axpos(2);
        set([gui.backax,gui.ax],'position',axpos)
    end
    
    gTemp=[];
    gTemp(1)=axpos(1)+axpos(3)-0.02;
    gTemp(2)=axpos(2)+axpos(4)-0.025;
    gTemp(3)=0.02;
    gTemp(4)=0.025;
    set(gui.MainFigCapture,'position',gTemp)
    gTemp(3)=0.04; gTemp(1)=gTemp(1)-0.04;
    set(gui.MainFigSave,'position',gTemp);
    gTemp(3)=0.06; gTemp(1)=gTemp(1)-0.06;
    set(gui.MainFigPause,'position',gTemp);
end

set(gui.fig,'userdata',gui)


%Figure DeleteFCN for cleanup
function DelFcn(varargin)
gui=get(findobj('tag','gSS07'),'userdata');
delete(findobj('tag','gSS07anal'))
delete(findobj('tag','gFullScreen'));
delete(findobj('tag','gSS07meta'))
delete(timerfind)
try
    stop(gui.ai)
    stop(gui.ao)
    delete(gui.ai)
    try stop(gui.sound); delete(gui.sound); end
    warning on MATLAB:Axes:NegativeDataInLogAxis
end

%Open the current data directory in windows explorer
function openpwd(obj,event)
dos(['explorer ',get(gcbo,'tag')]);

function LaunchManual(obj,event)
open SpikeHound-v1p0b-Manual.pdf

function launchWeb(obj,event)
web http://crawdad.cornell.edu/gprime/ -browser

function launchRef(obj,event)
web http://hdl.handle.net/1813/7530 -browser

function OpenMeta(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

if strcmpi(get(obj,'type'),'figure')
    set(gui.RecMeta,'value',0)
    gTemp=findobj('tag','gSS07meta');
    set(gui.RecMeta,'userdata',get(gui.ExperimentText,'string'));
    delete(obj)
    return
end

switch get(obj,'value')
    case 0
        gTemp=findobj('tag','gSS07meta');
        set(gui.RecMeta,'userdata',get(gui.ExperimentText,'string'));
        delete(gTemp)
        return
    case 1
        pos=get(gui.fig,'position');
        gTemp=figure('tag','gSS07meta','deletefcn',@OpenMeta,'position',...
            [pos(1)+810 pos(2) 300 600],'menubar','none','numbertitle','off','name','Experiment Details');
        gui.ExperimentText=uicontrol('style','edit','backgroundcolor','w','horizontalalignment','left','max',2,...
            'units','normalized','position',[0.02 0.02 0.96 0.96],'tag','metatext');        
        set(gui.ExperimentText,'string',get(gui.RecMeta,'userdata'));
end
set(gui.fig,'userdata',gui)

function StimSwitch(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
set(gui.OutputState,'value',0)
set(obj,'value',1)

switch obj
    case gui.OutputState(1) %Pulse Trains
        set([gui.FunctionGenerator gui.StimControls],'visible','off')
        set(gui.StimControls,'visible','on')        
    case gui.OutputState(2) %Function Generator
        set([gui.FunctionGenerator gui.StimControls],'visible','off')
        set(gui.FunctionGenerator,'visible','on')
end



%Extend Scope to fullscreen
function FullScreenScope(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
offFig=figure('tag','gFullScreen','menubar','none','numbertitle','off');
set(offFig,'name','Spike Hound Mobile Scope')
set(offFig,'windowbuttonupfcn',...
    'set(gcf,''windowbuttonmotionfcn'',''''); spikeHound(''ChangeLevel'');')
set(offFig,'DeleteFcn','spikeHound(''FullScreenCancel'')')
set(offFig,'WindowScrollWheelFcn',@ScrollScopeScale)

pos=[0.05 0.02 0.94 0.9];

set(gui.backax,'parent',offFig,'position',pos)
set(gui.ax,'parent',offFig,'position',pos)
set([gui.MainFigCapture gui.MainFigSave gui.MainFigPause],'parent',offFig)
set(gui.MainFigCapture,'position',[.98 .975 .02 .025])
set(gui.MainFigSave,'position',[.94 .975 .04 .025])
set(gui.AdvancedMenu(6),'enable','off')
set(gui.MainFigPause,'position',[.9 .975 .04 .025])


Context=get(gui.MainFigCapture,'UIContextMenu');
set(Context,'parent',offFig)
set(gui.MainFigCapture,'UIContextMenu',Context)


function FullScreenCancel
gui=get(findobj('tag','gSS07'),'userdata');
set(gui.backax,'parent',gui.fig,'position',[0.04 0.45 0.6 0.5])
set(gui.ax,'parent',gui.fig,'position',[0.04 0.45 0.6 0.5])
set([gui.MainFigCapture gui.MainFigSave gui.MainFigPause],'parent',gui.fig)
set(gui.MainFigCapture,'position',[0.62 0.925 0.02 0.025])
set(gui.MainFigSave,'position',[0.58 0.925 0.04 0.025])
set(gui.MainFigPause,'position',[0.52 0.925 0.06 0.025])
set(gui.AdvancedMenu(6),'enable','on')
set(findobj('tag','gFullScreen'),'tag','')

if get(gui.DIOConnect,'value')==1
    set([gui.backax,gui.ax],'position',[0.04 0.52 0.6 0.43]);
else
    set([gui.backax,gui.ax],'position',[0.04 0.45 0.6 0.5]);
end

Context=get(gui.MainFigCapture,'UIContextMenu');
set(Context,'parent',gui.fig)
set(gui.MainFigCapture,'UIContextMenu',Context)
figResize(gui.fig,'')

function multiSelect(obj, event)
% Handle The Multi-board and DIO Selector Boxes
gui=get(gcbf,'userdata');
set([gui.multiBoard,gui.multiChan,gui.multiDIO,gui.multiStim gui.multiNIDAQ],'value',0)
set(obj,'value',1)
set([gui.BoardConfigPanel,gui.ChannelAddPanel, gui.NIDAQConfigPanel,...
    gui.DigitalConfigPanel, gui.OutputConfigPanel],'visible','off')

if get(gui.multiBoard,'value')
    set(gui.BoardConfigPanel,'visible','on')
end
if get(gui.multiChan,'value')
    set(gui.ChannelAddPanel,'visible','on')
end
if get(gui.multiDIO,'value')
    set(gui.DigitalConfigPanel,'visible','on')
end
if get(gui.multiStim,'value')
    set(gui.OutputConfigPanel,'visible','on')
end
if get(gui.multiNIDAQ,'value')
    set(gui.NIDAQConfigPanel,'visible','on')
end

% Called when the user selects an interface board
function BoardSelect(obj,event,interfaceID,deviceID,devInd)
gui=get(findobj('tag','gSS07'),'userdata');
set(findobj('tag','gInMenu'),'checked','off');
set(obj,'checked','on')

set(gui.ChanDeviceName,'string',get(obj,'label'))

try
    stop(gui.ai)
    delete(gui.ai)
end

%Create AI
gui.ai=analoginput(interfaceID,deviceID);

set(gui.axtxt,'string',get(obj,'label'))
set(gui.TrigMan,'userdata',gui.ai)
try gui.ai.inputtype='singleended'; end

gTemp=daqhwinfo(gui.ai);

%Setup New Advanced Board Config
set(gui.BoardName,'string',get(gcbo,'label'))
set(gui.BoardDriver,'string',[gTemp.VendorDriverDescription,' ',gTemp.VendorDriverVersion])
if strcmpi(gTemp.AdaptorName,'winsound')
    set(gui.BoardInputType,'string',{'AC-Coupled'},'value',1,'enable','on')
else
    set(gui.BoardInputType,'string',{'Differential','SingleEnded'},'value',2,'enable','on')
end
set(gui.BoardInClockSource,'string',set(gui.ai,'ClockSource'),'enable','on')
set(gui.BoardSkewRate,'string',set(gui.ai,'ChannelSkewMode'),'enable','on')

%Extract NIDAQ Specific Settings
try
    s=set(gui.ai,'HwDigitalTriggerSource');
    for i=1:length(s)
        if strcmpi(s{i},get(gui.ai,'HwDigitalTriggerSource')); break; end
    end
    set(gui.NIDAQHWTS,'string',s,'value',i,'enable','on')
catch
   set(gui.NIDAQHWTS,'enable','off')
end

try
    s=set(gui.ai,'ExternalSampleClockSource');
    for i=1:length(s)
        if strcmpi(s{i},get(gui.ai,'ExternalSampleClockSource')); break; end
    end
    set(gui.NIDAQESaCS,'string',s,'value',i,'enable','on')
catch
    set(gui.NIDAQESaCS,'enable','off')
end
try
    s=set(gui.ai,'ExternalScanClockSource');
    for i=1:length(s)
        if strcmpi(s{i},get(gui.ai,'ExternalScanClockSource')); break; end
    end        
    set(gui.NIDAQEScCS,'string',s,'value',i,'enable','on')
catch
    set(gui.NIDAQEScCS,'enable','off')
end



for i=gui.AIControls
    a=get(i,'children');
    b=findobj(a,'type','axes');
    for j=1:length(b)
        a(a==b(j))=[];
    end
    set(a,'enable','on')
end

set(gui.fig,'userdata',gui)
ListChans
ChangeTrig

%See if the same device is an analoginput.  If so, provide option to connect to it
gFoo=daqhwinfo(interfaceID);
if ~strcmpi(event,'Loading')
    if ~isempty(gFoo.ObjectConstructorName{devInd,2})
        gTemp=questdlg('Connect to Analog Output on This Board?','Analog Output',...
            'Yes','No','Yes');
        if strcmpi(gTemp,'No')|isempty(gTemp)
            return
        end
        AOobj=findobj('type','uimenu','tag','gOutMenu','label',get(obj,'label'));
        for i=1:length(AOobj)
            gTemp=get(AOobj(i),'callback');
            if gTemp{4}==devInd
                AOobj=AOobj(i); break; 
            end
        end
        OutputBoardSelect(AOobj,'',interfaceID,deviceID,devInd)
    end
end

%Connect to the Analog Output if the user wishes to use a different board
function OutputBoardSelect(obj,event,interfaceID,deviceID,devInd)
gui=get(findobj('tag','gSS07'),'userdata');
try delete(gui.ao); end
gui.ao=[];
set(findobj('tag','gOutMenu'),'checked','off');
set(obj,'checked','on')

gui.ao=analogoutput(interfaceID,deviceID);
set([gui.StimModeTrig gui.StimModeStart],'userdata',gui.ao)
gTemp=daqhwinfo(gui.ao);
addchannel(gui.ao,gTemp.ChannelIDs(1:2));
set(gui.StimAmp,'string',num2str(max(gui.ao.channel(1).outputrange)),'userdata',...
    max(gui.ao.channel(1).outputrange))
set(gui.OutputCS,'string',set(gui.ao,'ClockSource'),'enable','on')

sz=size(gTemp.OutputRanges); s={};
nowOR=gui.ao.channel(1).outputrange;

for i=1:sz(1)
    s{i}=num2str(gTemp.OutputRanges(i,:));
    if gTemp.OutputRanges(i,1)==nowOR(1)&gTemp.OutputRanges(i,2)==nowOR(2)
        ind=i;
    end
end
set(gui.OutputRange,'string',s,'value',ind,'enable','on');

try 
    gui.ao.TransferMode='Interrupts'; 
catch
disp('Interrupts Transfer Mode Not Available for Output (Probably a good thing)'); 
end
try gui.ao.TransferMode='SingleDMA'; end
try gui.ao.TransferMode='DualDMA'; end

try
    s=set(gui.ao,'ExternalClockSource');
    for i=1:length(s)
        if strcmpi(s{i},get(gui.ao,'ExternalClockSource')); break; end
    end          
    set(gui.OutputECS,'string',s,'value',i,'enable','on')
catch
    set(gui.OutputECS,'enable','off','string',{' '},'value',1)
end
try
    s=set(gui.ao,'HwDigitalTriggerSource');
    for i=1:length(s)
        if strcmpi(s{i},get(gui.ao,'HwDigitalTriggerSource')); break; end
    end  
    set(gui.OutputHWTS,'string',s,'value',i,'enable','on')
catch
    set(gui.OutputHWTS,'enable','off','string',{' '},'value',1)
end

a=get(gui.StimControls,'children');
b=findobj(a,'type','axes');
for j=1:length(b)
    a(a==b(j))=[];
end
set(a,'enable','on')
set(gui.StimSRate,'enable','on')
set(gui.OutputState,'enable','on')

set(gui.fig,'userdata',gui)
StimParam



function BoardProperty(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
stop(gui.ai)
try; stop(gui.ao); end
switch obj
    case gui.NIDAQHWTS  %InHwDigitalTriggerSource
        s=get(obj,'string');
        set(gui.ai,'HwDigitalTriggerSource',s{get(obj,'value')})
    case gui.NIDAQESaCS %InExternalSampleClockSource
        s=get(obj,'string');
        set(gui.ai,'ExternalSampleClockSource',s{get(obj,'value')})
    case gui.NIDAQEScCS %InExternalScanClockSource
        s=get(obj,'string');
        set(gui.ai,'ExternalScanClockSource',s{get(obj,'value')})
    case gui.OutputECS  %OutExternalClockSource
        s=get(obj,'string');
        set(gui.ao,'ExternalClockSource',s{get(obj,'value')})
    case gui.OutputHWTS %OutHWDigitalTriggerSource
        s=get(obj,'string');
        set(gui.ai,'HWDigitalTriggerSource',s{get(obj,'value')})
    case gui.OutputCS %ClockSource for Output
        s=get(obj,'string');
        set(gui.ao,'ClockSource',s{get(obj,'value')});
    case gui.BoardInputType
        s=get(obj,'string');
        set(gui.ai,'InputType',s{get(obj,'value')});
    case gui.BoardInClockSource
        s=get(obj,'string');
        set(gui.ai,'ClockSource',s{get(obj,'value')});
    case gui.BoardSkewRate
        s=get(obj,'string');
        set(gui.ai,'ChannelSkewMode',s{get(obj,'value')});
    case gui.OutputRange
        s=get(obj,'string');
        range=str2num(s{get(obj,'value')});
        gui.ao.channel.outputrange=range;
end
StimParam
ChangeTrig

function digitalConnect(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
dIO=get(gui.DIOSelect,'userdata');
val=get(gui.DIOSelect,'value');

if get(gui.DIOLog,'value')
    set(gui.DIOLog,'value',0)
    DIOLogRun(gui.DIOLog,'')
end

if get(obj,'value')==0
        try stop(gui.dio); delete(gui.dio); end
        set(gui.DIOSelect,'enable','on')
        set(gui.DIOLines,'string',{' '},'enable','off','value',1)
        set([gui.DIOaddIn, gui.DIOaddOut, gui.DIORate, gui.DIOFile, gui.DIOLog, gui.DIOLogChange, gui.DIOLogLink],'enable','off')
        pos=[0.04 0.45 0.6 0.5];
        if get(gui.backax,'parent')==gui.fig; set([gui.backax, gui.ax],'position',pos); end
        delete(findobj('tag','gSS07dio'))
        figResize(gui.fig,'')
        return
end
pos=[0.04 0.52 0.6 0.43];
if get(gui.backax,'parent')==gui.fig; set([gui.backax, gui.ax],'position',pos); end
set(gui.backax,'ytick',linspace(min(get(gui.backax,'ylim')),max(get(gui.backax,'ylim')),11))

set(gui.DIOSelect,'enable','off')
gui.dio=digitalio(dIO{val,1},dIO{val,2});

gTemp=daqhwinfo(gui.dio);
s={};
ddir={};
for i=1:length(gTemp.Port)
    s{end+1}=['Entire Port',num2str(gTemp.Port(i).ID),' ',gTemp.Port(i).Direction];
    ddir{end+1,1}=gTemp.Port(i).Direction;
    ddir{end,2}=gTemp.Port(i).ID;
    ddir{end,3}=gTemp.Port(i).LineIDs;
    for p=1:length(gTemp.Port(i).LineIDs)
        s{end+1}=['P',num2str(gTemp.Port(i).ID),' Line:',num2str(gTemp.Port(i).LineIDs(p))];
        ddir{end+1,1}=gTemp.Port(i).Direction;
        ddir{end,2}=gTemp.Port(i).ID;
        ddir{end,3}=gTemp.Port(i).LineIDs(p);
    end
end
if isempty(s); s={' '}; end
set(gui.DIOLines,'string',s,'enable','on','userdata',ddir)
if strfind(gTemp.Port(1).Direction,'in'); set(gui.DIOaddIn,'enable','on'); end
if strfind(gTemp.Port(1).Direction,'out'); set(gui.DIOaddOut,'enable','on'); end
set([gui.DIORate,gui.DIOFile,gui.DIOLogChange,gui.DIOLog,gui.DIOLogLink],'enable','on')

set(gui.fig,'userdata',gui)

%Setup Timer and start(dio)
gui.dio.TimerPeriod=0.1;
gui.dio.TimerFcn=@UpdateDIO;

figResize(gui.fig,'')

function selectDIOport(obj,event)
%Display I/O support for a selected port pin
gui=get(findobj('tag','gSS07'),'userdata');
ddir=get(obj,'userdata');
set([gui.DIOaddIn,gui.DIOaddOut],'enable','off')
try
    ddir=ddir{get(obj,'value'),1};
    if strfind(ddir,'in'); set(gui.DIOaddIn,'enable','on'); end
    if strfind(ddir,'out'); set(gui.DIOaddOut,'enable','on'); end
end    

function addDIOport(obj,event)
%Add a line to the digitalio
gui=get(findobj('tag','gSS07'),'userdata');
stop(gui.dio)
direction=get(obj,'string');
ddir=get(gui.DIOLines,'userdata');
s=get(gui.DIOLines,'string');
index=get(gui.DIOLines,'value');
PortID=ddir{index,2};
LineID=ddir{index,3};
LineName=s{index};

if strfind(LineName,'Entire Port')
   for i=(index+1):length(s)
       if ddir{index+1,2}==ddir{index,2};
           addline(gui.dio,ddir{index+1,3},ddir{index+1,2},direction,s{index+1});
           s(index+1)=[];
           ddir(index+1,:)=[];
       end
   end
   s(index)=[];
   ddir(index,:)=[];
else
    addline(gui.dio,LineID,PortID,direction,LineName);
    s(index)=[];
    ddir(index,:)=[];
end

if isempty(s); s={' '}; end
if index>length(s); index=1; end
set(gui.DIOLines,'userdata',ddir,'string',s,'value',index)
set(gui.fig,'userdata',gui)
selectDIOport(gui.DIOLines,'')
buildDIOgui(gui);
for i=1:length(gui.dio.Line)
    if strcmpi(gui.dio.Line(i).Direction,'in'); start(gui.dio); return; end
end

function buildDIOgui(gui)
%Construct Buttons and such for Displaying the Digital States in the UI

delete(findobj('tag','gSS07dio'))
% set([gui.backax, gui.ax],'position',[0.04 0.45 0.6 0.5])
% set([gui.backax, gui.ax],'position',[0.04 0.52 0.6 0.43])
for i=1:length(gui.dio.Line)
    pos=[0.04+(i-1)*0.6/(length(gui.dio.Line)) 0.45 0.6/(length(gui.dio.Line)) 0.05];
    pos(1)=pos(1)*800; pos(3)=pos(3)*800; pos(2)=pos(2)*600; pos(4)=pos(4)*600;
    gui.dioport(i)=uicontrol('parent',gui.fig,'style','toggle','tag','gSS07dio',...
        'units','pixels','position',pos,...
        'userdata',gui.dio.Line(i),'callback',@DIOOutput);
end

states=getvalue(gui.dio);
for i=1:length(gui.dio.Line)
    obj=gui.dioport(i);
    set(obj,'string',[num2str(gui.dio.Line(i).Port),'.',num2str(gui.dio.Line(i).HwLine)])
    if states(i)==0
        set(obj,'backgroundcolor',get(gui.fig,'color'))
    else
        set(obj,'backgroundcolor','r')
    end
    if strcmpi(gui.dio.Line(i).Direction,'Out')
        set(obj,'enable','on')
    else
        set(obj,'enable','off')
    end        
end
gui.dio.TimerFcn={@UpdateDIO,gui.dioport};
set(gui.fig,'userdata',gui)

function DIOOutput(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

putvalue(get(obj,'userdata'),get(obj,'value'))

switch get(obj,'value')
    case 0
        set(obj,'backgroundcolor',get(gcf,'color'))
    case 1
        set(obj,'backgroundcolor','r')
end

function UpdateDIO(obj,event,dioport)
%Poll DIO Input for Values
% gui=get(findobj('tag','gSS07'),'userdata');
states=getvalue(obj);
for i=1:length(states)
    try set(dioport(i),'value',states(i),'backgroundcolor',[.8 .8 .8]*(states(i)==0)+[1 0 0]*states(i)); end
end

function DIOLogFile(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

[fname, pname] = uiputfile([datestr(now,30),'.txt'], 'Digital Input Log Filename',...
       [get(gui.DataMenu(1),'userdata'),datestr(now,30),'.txt']);

if isequal(fname,0) | isequal(pname,0)
   return
else
   set(gui.DIOFile,'string',fname);
   set(gui.DIOFile,'tag',fullfile(pname,fname));
end

set(gui.DataMenu(1),'userdata',pname);

function DIOLogRun(obj,event)
%Activate/Deactivate the DIO Logging Session
gui=get(findobj('tag','gSS07'),'userdata');

try stop(gui.dio); end
ins=0;
for i=1:length(gui.dio.Line)
    if strcmpi(gui.dio.Line(i).Direction,'in'); ins=1; end
end
if ins==0; set(obj,'value',0); end

switch get(obj,'value')
    case 0
        %Deactivate Log Session, Enable Controls
        set(obj,'backgroundcolor',[1 .7 .7],'string','Start Logging')
        stop(gui.dio)
        try 
            s=gui.dio.TimerFcn;
            fclose(s{2});
            stop(s{5});
            delete(s{5});
        end
        gui.dio.TimerPeriod=0.1;
        gui.dio.TimerFcn={@UpdateDIO,gui.dioport};
        for i=1:length(gui.dio.Line)
            if strcmpi(gui.dio.Line(i).Direction,'in'); start(gui.dio); break; end
        end
        set([gui.DIOConnect, gui.DIOLines, gui.DIOaddIn, gui.DIOaddOut,gui.DIORate,gui.DIOFile,gui.DIOLogChange],'enable','on')
        selectDIOport(gui.DIOLines,' ')
    case 1
        %Open File, Set Period, Set TimerFcn, Start DIO, Disable Controls
        set(obj,'backgroundcolor',[.7 1 .7],'string','Stop')
        if strcmpi(event,'RecSync')
            fname=gui.ai.LogFileName;
            fname=[fname(1:(end-4)),'_dio.txt'];
            f=fopen(fname,'w');
            t0=clock;
        else
            fname=get(gui.DIOFile,'tag');
            f=fopen(fname,'w');
            t0=clock;
        end
        fwrite(f,num2str(t0));
        fprintf(f,'\n');
        set(gui.dio,'Userdata',getvalue(gui.dio))
        set([gui.DIOConnect, gui.DIOLines, gui.DIOaddIn, gui.DIOaddOut,gui.DIORate,gui.DIOFile,gui.DIOLogChange],'enable','off')
        %setup a timer to update the ports in the graphics at .1s period
        t=timer('TimerFcn',{@DIOLogGraphic,gui.dio, gui.dioport},'Period',0.1,'BusyMode','queue',...
            'ExecutionMode','fixedRate','TaskstoExecute',inf);
        start(t)
        gui.dio.TimerPeriod=1/get(gui.DIORate,'userdata');
        gui.dio.TimerFcn={@DIOLogTimerFcn,f,get(gui.DIOLogChange,'value'),gui.dioport,t,t0};
        start(gui.dio)
end

function DIOLogTimerFcn(obj,event,f,logchange,dioport,t,t0)
%Function to log digital data to a file at a software sample rate
vals=getvalue(obj);
time=etime(clock,t0);
if logchange
    preval=get(obj,'userdata');
    if sum(vals~=preval)>0
        fprintf(f,'%0.5f ',time);
        fprintf(f,'%0.0f ',vals);
        fprintf(f,'\n');
    end
        set(obj,'userdata',vals)
else
    fprintf(f,'%0.5f ',time);
    fprintf(f,'%0.0f ',vals);
    fprintf(f,'\n');
end

function DIOLogGraphic(obj,event,dio,dioport)
%value is stored in dio userdata
states=getvalue(dio);
for i=1:length(states)
    set(dioport(i),'value',states(i),'backgroundcolor',[.8 .8 .8]*(states(i)==0)+[1 0 0]*states(i))
end

function SaveGPRIMEState(obj,event)
%Save Current Connection State
gui=get(findobj('tag','gSS07'),'userdata');

%General Scope Settings
settings.StreamScope=get(gui.AdvancedMenu(7),'checked');
settings.FigurePosition=get(gui.fig,'position');
settings.FullScreenScope=get(gui.AdvancedMenu(6),'checked');
settings.TimeFullSpan=get(gui.Refresh,'value');
settings.listen=get(gui.ChanAudio,'value');

%Input Interface
menuobj=findobj('tag','gInMenu','checked','on');
if isempty(menuobj)
    settings.board.InState=0;
else
    settings.board.InState=1;
    settings.board.fC=get(menuobj,'callback');
    settings.board.sRate=gui.ai.samplerate;
    settings.board.InputType=gui.ai.InputType;
    settings.board.InClockSource=gui.ai.ClockSource;
    settings.board.ChannelSkewMode=gui.ai.ChannelSkewMode;
    try settings.board.HwDigitalTriggerSource=gui.ai.HwDigitalTriggerSource; end
    try settings.board.ExternalSampleClockSource=gui.ai.ExternalSampleClockSource; end
    try settings.board.ExternalScanClockSource=gui.ai.ExternalScanClockSource; end
    settings.board.chans=length(gui.ai.Channel);

    %Channel Config
    for i=1:length(gui.ai.Channel)
        Chan=get(gui.ChanControls(i),'userdata');
        settings.board.channels(i).ID=gui.ai.Channel(i).HwChannel;
        settings.board.channels(i).Name=gui.ai.Channel(i).ChannelName;
        settings.board.channels(i).InputRange=get(Chan.VR,'value');
        settings.board.channels(i).ExternalGain=get(Chan.extG,'userdata');
        settings.board.channels(i).VpD=get(Chan.VpD,'value');
        settings.board.channels(i).Display=get(Chan.Show,'value');
        settings.board.channels(i).SubtractDC=get(Chan.ACCouple,'value');
        settings.board.channels(i).Color=get(gui.ChanControls(i),'backgroundcolor');
        settings.board.channels(i).Offset=get(Chan.Offset,'userdata');
    end
end

%Output Interface Configurations
menuobj=findobj('tag','gOutMenu','checked','on');
if isempty(menuobj)
    settings.board.OutState=0;
else
    settings.board.OutState=1;
    settings.out.fC=get(menuobj,'callback');
    
    try settings.out.ExternalClockSource=gui.ao.ExternalClockSource; end
    try settings.out.HwDigitalTriggerSource=gui.ao.HwDigitalTriggerSource; end
    try settings.out.ClockSource=gui.ao.ClockSource; end
    settings.out.sRate=gui.ao.SampleRate;

    settings.PulseTrain.StimMode=get(gui.StimMode,'value'); %Cell array
    settings.PulseTrain.Type=get(gui.StimType,'value'); %Cell array
    settings.PulseTrain.Shape=get(gui.StimShape,'userdata'); %Cell array
    settings.PulseTrain.StimTetDur=get(gui.StimTetDur,'userdata');
    settings.PulseTrain.StimTetInt=get(gui.StimTetInt,'userdata');
    settings.PulseTrain.StimRepeat=get(gui.StimRepeat,'userdata');
    settings.PulseTrain.StimAmp=get(gui.StimAmp,'userdata');
end
    
    
%Digital I/O State
settings.dio.DIOConnect=get(gui.DIOConnect,'value');
if settings.dio.DIOConnect==1
    gTemp=daqhwinfo(gui.dio);
    s=get(gui.DIOSelect,'userdata');
    val=get(gui.DIOSelect,'value');
    settings.dio.Adaptor=s{val,1};
    settings.dio.ID=s{val,2};
    settings.dio.numlines=length(gui.dio.Line);
    for i=1:length(gui.dio.Line)
        settings.dio.line(i).Port=gui.dio.Line(i).Port;
        settings.dio.line(i).HwLine=gui.dio.Line(i).HwLine;
        settings.dio.line(i).direction=gui.dio.Line(i).Direction;
    end
end

[fname, pname] = uiputfile('default.mat', 'Save Scope State (default.mat will auto-load)',...
       [get(gui.DataMenu(1),'userdata'),'.mat']);

if isequal(fname,0) | isequal(pname,0)
   return
else
   save(fullfile(pname,fname),'settings')
end

set(gui.DataMenu(1),'userdata',pname);


function LoadGPRIMEState(obj,event)
%Load a Previously saved Scope State
gui=get(findobj('tag','gSS07'),'userdata');
if strcmpi(event,'default')
    load('default.mat')
else
[fname, pname] = uigetfile('*.mat','Load Scope Settings',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
set(gui.DataMenu(1),'userdata',pname)
load(fullfile(pname,fname))
end

%Reset Interface
daqreset

% Configure AI
if settings.board.InState==1
    obj=findobj('callback',settings.board.fC);
    set(gui.ChanAudio,'value',0)
    fC=settings.board.fC;
    BoardSelect(obj,'Loading',fC{2},fC{3},fC{4})

    gui=get(findobj('tag','gSS07'),'userdata');
    %Setup Board config
    gui.ai.samplerate=settings.board.sRate;
    set(gui.SRate,'string',num2str(gui.ai.samplerate),'userdata',gui.ai.samplerate)
    gui.ai.InputType=settings.board.InputType;
    s=get(gui.BoardInputType,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.InputType); set(gui.BoardInputType,'value',i); end; end
    gui.ai.ClockSource=settings.board.InClockSource;
    s=get(gui.BoardInClockSource,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.ClockSource); set(gui.BoardInClockSource,'value',i); end; end
    gui.ai.ChannelSkewMode=settings.board.ChannelSkewMode;
    s=get(gui.BoardSkewRate,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.ChannelSkewMode); set(gui.BoardSkewRate,'value',i); end; end
    try gui.ai.HWDigitalTriggerSource=settings.board.HwDigitalTriggerSource; 
    s=get(gui.NIDAQHWTS,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.HWDigitalTriggerSource); set(gui.NIDAQHWTS,'value',i); end; end
    end
    try gui.ai.ExternalSampleClockSource=settings.board.ExternalSampleClockSource; 
    s=get(gui.NIDAQESaCS,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.ExternalSampleClockSource); set(gui.NIDAQESaCS,'value',i); end; end
    end
    try gui.ai.ExternalScanClockSource=settings.board.ExternalScanClockSource; 
    s=get(gui.NIDAQEScCS,'string');
    for i=1:length(s); if strcmpi(s{i},gui.ai.ExternalScanClockSource); set(gui.NIDAQEScCS,'value',i); end; end
    end

    %Addchannels
    for i=1:length(settings.board.channels)
        set(gui.ChanColorSelect,'backgroundcolor',settings.board.channels(i).Color)
        set(gui.ChanNameSet,'string',settings.board.channels(i).Name)
%         s=get(gui.ChanList,'string');
        s=get(gui.AvailableList,'string');
        for j=1:length(s); 
            if settings.board.channels(i).ID==str2num(s{j}); 
                set(gui.AvailableList,'value',j); 
            end; 
        end
        ChannelAdd(1,'')
        gui=get(findobj('tag','gSS07'),'userdata');
        %Setup display/gain settings
        %InputRange,ExternalGain,VpD,Display,SubtractDC
        Chan=get(gui.ChanControls(end),'userdata');
        set(Chan.extG,'string',num2str(settings.board.channels(i).ExternalGain),'userdata',settings.board.channels(i).ExternalGain)
        set(Chan.Offset,'string',num2str(settings.board.channels(i).Offset),'userdata',settings.board.channels(i).Offset)
        set(Chan.VpD,'value',settings.board.channels(i).VpD)
        set(Chan.extG,'string',num2str(settings.board.channels(i).ExternalGain),'userdata',settings.board.channels(i).ExternalGain)
        set(Chan.Show,'value',settings.board.channels(i).Display)
        set(Chan.ACCouple,'value',settings.board.channels(i).SubtractDC)
        set(Chan.VR,'value',settings.board.channels(i).InputRange)
        s=get(Chan.VR,'string');
        stop(gui.ai)
        set(gui.ai.Channel(end),'inputrange',str2num(s{get(Chan.VR,'value')}));    
        scaleAx
        drawnow
    end
end

%Configure AO
if settings.board.OutState==1
    obj=findobj('callback',settings.out.fC);
    fC=settings.out.fC;    
    OutputBoardSelect(obj,'Loading',fC{2},fC{3},fC{4})
    gui=get(findobj('tag','gSS07'),'userdata');
    
    try
        try gui.ao.ExternalClockSource=setting.out.ExternalClockSource; 
           s=get(gui.OutputECS,'string');
           for i=1:length(s); if strcmpi(s{i},gui.ao.ExternalClockSource); set(gui.OutputECS,'value',i); end; end
        end
        try gui.ao.HwDigitalTriggerSource=settings.out.HwDigitalTriggerSource;
            s=get(gui.OutputHWTS,'string');
            for i=1:length(s); if strcmpi(s{i},gui.ao.HwDigitalTriggerSource); set(gui.OutputHWTS,'value',i); end; end
        end
        gui.ao.ClockSource=settings.out.ClockSource;
        s=get(gui.OutputCS,'string');
        for i=1:length(s); if strcmpi(s{i},gui.ao.ClockSource); set(gui.OutputCS,'value',i); end; end
        gui.ao.SampleRate=settings.out.sRate;
        set(gui.StimSRate,'string',num2str(gui.ao.SampleRate),'userdata',gui.ao.SampleRate)
    end

    set(gui.StimType,{'value'},settings.PulseTrain.Type)
    for i=1:length(settings.PulseTrain.Shape)
        set(gui.StimShape(i),'userdata',settings.PulseTrain.Shape{i},'string',num2str(settings.PulseTrain.Shape{i}));
    end    
        set(gui.StimMode,{'value'},settings.PulseTrain.StimMode)
        set(gui.StimTetDur,'userdata',settings.PulseTrain.StimTetDur,'string',num2str(settings.PulseTrain.StimTetDur));
        set(gui.StimTetInt,'userdata',settings.PulseTrain.StimTetInt,'string',num2str(settings.PulseTrain.StimTetInt));
        set(gui.StimRepeat,'userdata',settings.PulseTrain.StimRepeat,'string',num2str(settings.PulseTrain.StimRepeat));
        set(gui.StimAmp,'userdata',settings.PulseTrain.StimAmp,'string',num2str(settings.PulseTrain.StimAmp));
end

%Set Scope States
s=get(gui.Refresh,'string');
set(gui.Refresh,'value',settings.TimeFullSpan,'userdata',str2double(s{settings.TimeFullSpan}))
set(gui.fig,'position',settings.FigurePosition)
set(gui.AdvancedMenu(7),'checked',settings.StreamScope)

drawnow
ChangeTrig
StimParam
scaleAx

%Connect to DIO
if settings.dio.DIOConnect==1
    s=get(gui.DIOSelect,'string');
    dIO=get(gui.DIOSelect,'userdata');
    for i=1:length(s)
        if strcmpi(dIO{i,1},settings.dio.Adaptor)&&strcmpi(dIO{i,2},settings.dio.ID)
            set(gui.DIOSelect,'value',i)
        end
    end
    set(gui.DIOConnect,'value',1)
    digitalConnect(gui.DIOConnect,'')
    gui=get(findobj('tag','gSS07'),'userdata');
    
    if settings.dio.numlines>0
        for i=1:settings.dio.numlines
            addline(gui.dio,settings.dio.line(i).HwLine,settings.dio.line(i).Port,settings.dio.line(i).direction);
        end
        buildDIOgui(gui)
        for i=1:length(gui.dio.Line)
            if strcmpi(gui.dio.Line(i).Direction,'in'); start(gui.dio); break; end
        end
    end
    gui=get(findobj('tag','gSS07'),'userdata');
    
    %clean up the selector UI w/ only available DIO port options
    s=get(gui.DIOLines,'string');
    ddir=get(gui.DIOLines,'userdata');
    deleteIDs=[];
    for i=1:length(s)
        for j=1:length(gui.dio.Line)
            if (ddir{i,2}==gui.dio.Line(j).Port)&(ddir{i,3}==gui.dio.Line(j).HwLine)
                deleteIDs=[deleteIDs,i];
            end
        end
    end
    s(deleteIDs)=[];
    ddir(deleteIDs,:)=[];
    set(gui.DIOLines,'string',s,'userdata',ddir)
end
figResize(gui.fig,[])


%Called to List the Channels Available on the Interface Device
function ListChans
gui=get(findobj('tag','gSS07'),'userdata');
gTemp=daqhwinfo(gui.ai);
stop(gui.ai)
delete(gui.ai.channel)
CleanChans
s=[];
for i=1:length(gTemp.SingleEndedIDs)
    s{i}=num2str(gTemp.SingleEndedIDs(i));
end
if isempty(s)
    for i=1:length(gTemp.DifferentialIDs)
        s{i}=num2str(gTemp.DifferentialIDs(i));
    end
end
set(gui.AvailableList,'string',s)

%Create a New Channel for the AI
function ChannelAdd(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

%Gather Channel Information and Create it
gTemp=get(gui.AvailableList,'string');
ChanID=str2double(gTemp{get(gui.AvailableList,'value')});
if isnan(ChanID); return; end
stop(gui.ai)

%If channel name is empty, set default channel name
ChanName=get(gui.ChanNameSet,'string');  
if isempty(ChanName); ChanName=['Chan',gTemp{get(gui.AvailableList,'value')}]; end

ChanColor=get(gui.ChanColorSelect,'backgroundcolor');

%Create Channel
gTempChan=addchannel(gui.ai,ChanID,ChanName);

%Add channel name to Active Channel List
if isempty(get(gui.ChanList,'string'))
    s{1}=ChanName;
    set(gui.ChanList,'string',s)
else
    s=get(gui.ChanList,'string');
    s{end+1}=ChanName;
    set(gui.ChanList,'string',s)
end

gTemp(get(gui.AvailableList,'value'))=[];
if isempty(gTemp); gTemp={' '}; end
set(gui.AvailableList,'string',gTemp)

%Create Graph Elements
try set(gui.ax,'visible','off'); end
gui.ax=[gui.ax,axes('position',get(gui.backax,'position'),...
    'parent',get(gui.backax,'parent'),'xgrid','off','ygrid','off',...
    'color','none','xtick',[],'tag',ChanName)];
set(gui.ax(end),'visible','on','ycolor',ChanColor,'userdata',0)
gui.pl=[gui.pl,line([0 1],[0 0],'color',ChanColor,'parent',gui.ax(end))];
gui.tx=[gui.tx,text(0,0,ChanName,'parent',gui.ax(end))];
set(gui.tx(end),'buttondownfcn',{@DragTrace,ChanName},...
    'fontweight','bold','tag',ChanName)

%Create Control Elements
fpos=get(gui.fig,'position');
pos=[521   145   272   120];
% pos(1)=fpos(3)-(800-521);
gui.ChanControls=[gui.ChanControls,...
    uipanel('title',[ChanName,' - HW:',num2str(ChanID)],...
    'units','pixels','position',pos,...
    'backgroundcolor',get(gcf,'color'),'tag',ChanName,'visible','off')];


%Display latest created channel controls
for i=1:length(gui.ChanControls); set(gui.ChanControls(i),'visible','off'); end 
set(gui.ChanControls(end),'visible','on'); 
set(gui.ChanList,'value',length(gui.ChanControls));
set(gui.ChanControls(end),'backgroundcolor',ChanColor)

%Intelligent font color selection
if mean(get(gui.ChanControls(end),'backgroundcolor'))<0.5
        set(gui.ChanControls(end),'foregroundcolor','w'); 
else
        set(gui.ChanControls(end),'foregroundcolor','k'); 
end

gTemp=daqhwinfo(gui.ai);
Chan.ID=ChanID;
set(gui.ChanNameSet,'string','');

uicontrol('parent',gui.ChanControls(end),'string','Vert Offset (%)','style','text',...
    'backgroundcolor',get(gui.ChanControls(end),'backgroundcolor'),'units',...
    'normalized','position',[0.01 0.38 0.38 0.15],'HorizontalAlignment','right','visible','off')
Chan.Offset=uicontrol('parent',gui.ChanControls(end),'style','edit','string','0',...
    'backgroundcolor','w','units','normalized','position',[.45 .38 .3 .2],...
    'userdata',0,'visible','off');


uicontrol('parent',gui.ChanControls(end),'string','Volts per Division','style','text',...
    'backgroundcolor',get(gui.ChanControls(end),'backgroundcolor'),'units',...
    'normalized','position',[0.45 0.8 0.38 0.15],'HorizontalAlignment','left')
Chan.VpD=uicontrol('parent',gui.ChanControls(end),'style','popupmenu','string',...
 {'20','10','5','2','1','0.5','0.2','0.1','0.05','0.02','0.01','0.005','0.001','0.0001','0.00005','0.00001'},...
    'backgroundcolor','w','units','normalized','position',[.1 .78 .3 .2],'tag','VpD',...
    'userdata',[20,10,5,2,1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.001,0.0001,0.00005,0.00001],...
    'value',6,'callback','spikeHound(''scaleAx'')');
uicontrol('parent',gui.ChanControls(end),'string','ADC Input Range','style','text',...
    'backgroundcolor',get(gui.ChanControls(end),'backgroundcolor'),'units',...
    'normalized','position',[0.01 0.45 0.38 0.15],'HorizontalAlignment','right')
Chan.VR=uicontrol('parent',gui.ChanControls(end),'style','popupmenu','string','',...
    'backgroundcolor','w','units','normalized','position',[.45 .45 .45 .2],'tag','VpD',...
    'userdata',1,'value',1,'userdata',gui.ai.channel(end),'callback',...
        ['stop(get(get(gcbo,''userdata''),''parent'')); ',...
        'set(get(gcbo,''userdata''),''inputrange'',str2num(popupstr(gcbo))); ',...
        'spikeHound(''ChangeTrig'');']);
uicontrol('parent',gui.ChanControls(end),'string','External Gain','style','text',...
    'backgroundcolor',get(gui.ChanControls(end),'backgroundcolor'),'units',...
    'normalized','position',[0.01 0.20 0.38 0.15],'HorizontalAlignment','right')
Chan.extG=uicontrol('parent',gui.ChanControls(end),'style','edit','string',...
 '1','backgroundcolor','w','units','normalized','position',[.45 .20 .3 .2],'tag','extG',...
    'userdata',1,'value',5,'callback',...
    ['if isnan(str2double(get(gcbo,''string''))); set(gcbo,''string'',num2str(get(gcbo,''userdata'')));',...
    'else set(gcbo,''userdata'',str2double(get(gcbo,''string''))); drawnow; end']);

Chan.Show=uicontrol('parent',gui.ChanControls(end),'style','Checkbox','string',...
    'Display Channel','backgroundcolor',get(gui.ChanControls(end),'backgroundcolor'),...
    'units','normalized','position',[0.02 0.02 .45 0.14],'value',1,'userdata',...
    gui.pl(end),'callback','spikeHound(''HidePlots'')');
Chan.ACCouple=uicontrol('parent',gui.ChanControls(end),'style','Checkbox','string',...
    'Subtract DC (Visually)','backgroundcolor',...
    get(gui.ChanControls(end),'backgroundcolor'),'units',...
    'normalized','position',[0.45 0.02 .55 0.14],'value',0,'callback',...
    'spikeHound(''scaleAx'');');

%Set an appropriate text color based on background color
chil=get(gui.ChanControls(end),'children');
for i=1:length(chil)
    if mean(get(chil(i),'backgroundcolor'))<0.5; set(chil(i),'foregroundcolor','w'); end
end
set(gui.ChanControls(end),'userdata',Chan)
set(gui.HWMenu,'userdata',1)

if length(gui.ChanControls)==1 
    set(gui.ChanColorSelect,'backgroundcolor',[.7 0 0])
elseif length(gui.ChanControls)==2
    set(gui.ChanColorSelect,'backgroundcolor',[0 0 0])
else
    set(gui.ChanColorSelect,'backgroundcolor',[0 0 .9])
end

s=size(gTemp.InputRanges);
irange=gui.ai.channel(end).InputRange;
for i=1:s(1)
    if irange(1)==gTemp.InputRanges(i,1)&&irange(2)==gTemp.InputRanges(i,2)
        tempVal=i;
    end
    str{i}=['[',num2str(gTemp.InputRanges(i,:)),']'];
end
set(Chan.VR,'string',str,'value',tempVal)

gTemp=get(gui.ChanList,'string');
if isempty(gTemp)
    gTemp=' ';
end

set(gui.TrigChan,'string',gTemp,...
    'callback','spikeHound(''ChangeTrig'')')
set(gui.TrigLevel,'callback',...
    ['if isnan(str2double(get(gcbo,''string''))); ',...
    'set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else; ',...
    'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end; ',...
    'spikeHound(''ChangeTrig'')']);

set(gui.fig,'userdata',gui)

if get(gui.ChanDispQ,'value')==0
    set(get(get(gui.pl(end),'parent'),'children'),'visible','off')
    set(Chan.Show,'value',0)
end    

%Keep the figure Extract button on top so user can right click it
if get(gui.MainFigCapture,'parent')==gui.fig
    gTemp=get(gui.fig,'children');
    gTemp(gTemp==gui.MainFigCapture)=[];
    gTemp=[gui.MainFigCapture; gTemp(:)];
    set(gui.fig,'children',gTemp)
end

scaleAx
initAI
if get(gui.DataAnalysisMode,'value')==1
    CleanChanLists
end

if length(gui.ai.channel)==0; 
    set(gui.DataAnalysisMode,'enable','off','value',0); 
    InitAnalysis(gui.DataAnalysisMode,[])
else
    set(gui.DataAnalysisMode,'enable','on')
end



%Remove a Single Channel from the AI
function ChannelRemove(obj,event)
gui=get(gcbf,'userdata');
if isempty(get(gui.ChanList,'string')); return; end
stop(gui.ai)

ind=get(gui.ChanList,'value');
Chan=get(gui.ChanControls(ind),'userdata');
delete(gui.ai.Channel(ind))
delete(gui.ChanControls(ind))
delete(gui.ax(ind))
gui.ChanControls(ind)=[]; gui.pl(ind)=[]; gui.ax(ind)=[]; gui.tx(ind)=[];
set(gui.fig,'userdata',gui)

ChanList=get(gui.ChanList,'string');
ChanList(ind)=[];
if get(gui.ChanList,'value')>length(ChanList)
    set(gui.ChanList,'string',ChanList,'value',length(ChanList))
else
    set(gui.ChanList,'string',ChanList)
end


if ~isempty(get(gui.ChanList,'string'));
    start(gui.ai)
else
    set(gui.ChanColorSelect,'backgroundcolor',[0 0 .9])
end

s=get(gui.AvailableList,'string');
s{end+1}=num2str(Chan.ID);
s=sort(s);
if isnan(str2double(s{1})); s(1)=[]; end
set(gui.AvailableList,'string',s)

gTemp=get(gui.ChanList,'string');
if isempty(gTemp)
    gTemp=' ';
end
set(gui.TrigChan,'string',gTemp,...
    'callback','spikeHound(''ChangeTrig'')')
set(gui.TrigLevel,'callback',...
    ['if isnan(str2double(get(gcbo,''string''))); ',...
    'set(gcbo,''string'',num2str(get(gcbo,''userdata''))); else; ',...
    'set(gcbo,''userdata'',str2double(get(gcbo,''string''))); end; ',...
    'spikeHound(''ChangeTrig'')']);

ChangeTrig

set(gui.fig,'userdata',gui)
DisplayChanControls
CleanChanLists

if length(gui.ai.channel)==0; 
    set(gui.DataAnalysisMode,'enable','off','value',0); 
    InitAnalysis(gui.DataAnalysisMode,[])
else
    set(gui.DataAnalysisMode,'enable','on')
end


%When a show button is clicked on a plot
function HidePlots
pl=get(gcbo,'userdata');
switch get(gcbo,'value')
    case 0
        set(get(get(pl,'parent'),'children'),'visible','off')
    case 1
        set(get(get(pl,'parent'),'children'),'visible','on')
end
gui=get(findobj('tag','gSS07'),'userdata');
for i=1:length(gui.pl)
    set(gui.pl(i),'xdata',[0],'ydata',[0],'userdata',[0])
end

%called when any individual channel VpD is changed
function scaleAx
gui=get(findobj('tag','gSS07'),'userdata');

Span=get(gui.Refresh,'string');
Span=str2double(Span(get(gui.Refresh,'value'),:));


for i=1:length(gui.pl)
    Chan=get(gui.ChanControls(i),'userdata');
    offset=str2double(get(Chan.Offset,'string')); 
    if isnan(offset); offset=get(Chan.Offset,'userdata'); end
    VpD=get(Chan.VpD,'string');
    VpD=str2double(VpD(get(Chan.VpD,'value')));
    offset=.01*offset*VpD*5;

    TempRange=VpD*[-5 5]-offset;
    set(gui.ax(i),'ylim',TempRange,'xlim',[0 1]*Span)
    switch get(Chan.ACCouple,'value')
        case 0
            raw=get(gui.pl(i),'userdata');
            set(gui.pl(i),'ydata',raw,'xdata',linspace(0,length(raw)/gui.ai.samplerate,length(raw)));
        case 1
            raw=get(gui.pl(i),'userdata');
            set(gui.pl(i),'ydata',raw-mean(raw),'xdata',linspace(0,length(raw)/gui.ai.samplerate,length(raw)));
    end
    switch get(Chan.Show,'value')
        case 0
            set(get(get(gui.pl(i),'parent'),'children'),'visible','off')
        case 1
            set(get(get(gui.pl(i),'parent'),'children'),'visible','on')
    end

end
set(gui.backax,'xlim',[0 1],'xtick',linspace(-1,1,21),'xticklabel',...
    round((linspace(-Span,Span,21)-...
    Span*get(gui.TriggerDelay,'userdata')/100)*5000)/5000,'ylim',[-1 1])



%Control Graphically Dragging a trace on screen
function DragTrace(obj,event,ChanName)
gui=get(findobj('tag','gSS07'),'userdata');
for i=1:length(gui.tx)
    if strcmpi(ChanName,get(gui.tx(i),'string'))
        ind=i; break;
    end
end
set(gcf,'windowbuttonmotionfcn',{@yMotionOffset,ChanName})

function yMotionOffset(obj,event,ChanName)
gui=get(findobj('tag','gSS07'),'userdata');
for i=1:length(gui.tx)
    if strcmpi(ChanName,get(gui.tx(i),'string'))
        ind=i; break;
    end
end
a=get(gui.backax,'currentpoint');
Chan=get(gui.ChanControls(i),'userdata');
ylim=get(gui.backax,'ylim');
if a(1,2)>max(ylim)||a(1,2)<min(ylim); return; end
set(Chan.Offset,'string',num2str(10*round(10*a(1,2)/max(ylim))),...
                'userdata',10*round(10*a(1,2)/max(ylim)))

offset=str2double(get(Chan.Offset,'string'));
VpD=get(Chan.VpD,'string');
VpD=str2double(VpD{get(Chan.VpD,'value')});
    
offset=.01*offset*VpD*5;
TempRange=VpD*[-5 5]-offset;
set(gui.ax(i),'ylim',TempRange)

%Left and right percentage based time alignment
function DragTime(obj,event)
set(gcf,'windowbuttonmotionfcn',@xMotionOffset)

function xMotionOffset(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

a=get(gui.backax,'currentpoint');
xlim=get(gui.backax,'xlim');

if a(1,1)>=(max(xlim)*.9)|a(1,1)<min(xlim); return; end
set(gui.TriggerDelay,'string',num2str(round(10*a(1,1)/max(xlim))*10),...
                     'userdata',round(10*a(1,1)/max(xlim))*10)
set(gui.timetext,'position',[round(get(gui.TriggerDelay,'userdata')/10)/10 1.15 0])
ChangeTrig


% Control Active Measurement
function MeasureGo
gui=get(findobj('tag','gSS07'),'userdata');
if isempty(get(gui.ChanList,'string')); return; end

set(gui.ax(get(gui.ChanList,'value')),'buttondownfcn','spikeHound(''MeasureDraw'')')
set(gui.pl(get(gui.ChanList,'value')),'buttondownfcn','spikeHound(''MeasureDraw'')')

function MeasureDraw
gui=get(findobj('tag','gSS07'),'userdata');
if ~isempty(findobj('tag','gSS07measure'))
    delete(findobj('tag','gSS07measure'))
    delete(findobj('tag','gSS07measureT'))
    set(gui.ax(get(gui.ChanList,'value')),'buttondownfcn','')
    set(gui.pl(get(gui.ChanList,'value')),'buttondownfcn','')
    return
end

measureax=gui.ax(get(gui.ChanList,'value'));
a=get(measureax,'currentpoint');
time=a(1,1);
level=a(1,2);

line([1 1]*time,[1 1]*level,'userdata',[time level],'tag','gSS07measure','color','k',...
    'linewidth',1,'marker','.','buttondownfcn','spikeHound(''reMeasureMotion'')',...
    'parent',measureax)
text(min(get(measureax,'xlim')),min(get(measureax,'ylim')),'','tag',...
    'gSS07measureT','horizontalalignment','left','parent',measureax,...
    'verticalalignment','bottom','fontweight','bold')
set(gcf,'windowbuttonmotionfcn',@MeasureMotion)

function MeasureMotion(obj,event)
gui=get(obj,'userdata');
trace=findobj('tag','gSS07measure');
Tlabel=findobj('tag','gSS07measureT');
anchor=get(trace,'userdata');
measureax=gui.ax(get(gui.ChanList,'value'));
a=get(measureax,'currentpoint');
time=a(1,1);
level=a(1,2);
set(trace,'xdata',[anchor(1) time],'ydata',[anchor(2) level])
dx=diff([anchor(1) time]);  dy=diff([anchor(2) level]); dr=abs(sqrt(dx^2+dy^2));    
dx=sprintf('%0.4f',dx); dy=sprintf('%0.4f',dy);  dr=sprintf('%0.4f',dr);
set(Tlabel,'string',['dx=',dx,'  dy=',dy,'  dr=',dr])
drawnow

function reMeasureMotion
set(gcf,'windowbuttonmotionfcn',@MeasureMotion)

%Move the threshold bar
function MoveThresh(obj,event,ChanName)
gui=get(findobj('tag','gSS07'),'userdata');
for i=1:length(gui.tx)
    if strcmpi(ChanName,get(gui.tx(i),'string'))
        ind=i; break;
    end
end

set(gcf,'windowbuttonmotionfcn',{@ThreshOffset,ChanName})

function ThreshOffset(obj,event,ChanName)
gui=get(findobj('tag','gSS07'),'userdata');
ax=get(findobj('tag','gSS07thresh'),'parent');
a=get(ax,'currentpoint');
ylim=get(ax,'ylim');
if a(1,2)>max(ylim)||a(1,2)<min(ylim); return; end
set(gui.TrigLevel,'string',num2str(round(10000*a(1,2))/10000),...
                'userdata',round(10000*a(1,2))/10000)
set(findobj('tag','gSS07thresh'),'ydata',[1 1]*get(gui.TrigLevel,'userdata'))



%When the user clicks on an active channel, bring it's controls up
function DisplayChanControls(varargin)
gui=get(findobj('tag','gSS07'),'userdata');

set(gui.ChanControls,'visible','off')
set(gui.ax,'visible','off')
ChanInd=get(gui.ChanList,'value');
if ChanInd==0; return; end
ax=gui.ax(ChanInd);
set(ax,'visible','on')
Chan=get(gui.ChanControls(ChanInd),'userdata');
if get(Chan.Show,'value'); set(get(ax,'children'),'visible','on'); end
set(gui.ChanControls(get(gcbo,'value')),'visible','on')

%Remove all channels from the analoginput
function CleanChans
gui=get(findobj('tag','gSS07'),'userdata');
stop(gui.ai)
delete(gui.ChanControls)
delete(gui.pl); delete(gui.tx);
delete(gui.ax); 
gui.ChanControls=[]; gui.pl=[]; gui.ax=[]; gui.tx=[];
set(gui.ChanList,'string','')
delete(gui.ai.Channel)
set(gui.ChanColorSelect,'backgroundcolor',[0 0 .9])
set(gui.fig,'userdata',gui)

%Called on mouse up and changes the levels for triggering
function ChangeLevel(varargin)
gui=get(findobj('tag','gSS07'),'userdata');
if nargin~=0
    if varargin{1}==gui.fig
        set(gcf,'windowbuttonmotionfcn','')
    end
end

if strcmpi(get(gui.TrigLevel,'enable'),'off'); return; end

if gui.ai.TriggerConditionValue==str2double(get(gui.TrigLevel,'string'));
    Span=str2double(popupstr(gui.Refresh));
    if (get(gui.Trig(3),'value')==1&&...
            gui.ai.TriggerDelay==-.01*get(gui.TriggerDelay,'userdata')*Span)||...
            get(gui.Trig(3),'value')==0;
        return
    end
end
    ChangeTrig

%Called when Trigger State Changes (or to reset AI)
function ChangeTrig(varargin)
gui=get(findobj('tag','gSS07'),'userdata');

if  get(gui.DIOLogLink,'value')&get(gui.DIOLog,'value')
    set(gui.DIOLog,'value',0)
    DIOLogRun(gui.DIOLog,'')
end

if gcbo==gui.TrigLevel
    lvl=str2double(get(gcbo,'string'));
    if isnan(lvl)
        lvl=get(gcbo,'userdata');
        set(gcbo,'string',num2str(lvl))
    else
        set(gcbo,'userdata',lvl)
    end
end

if ~isempty(gcbo)
    if sum(gcbo==gui.Trig)>0
        if nargin>2
        if strcmpi(varargin{3},'TrigRadio')
            set(get(gcbo,'userdata'),'value',0)
            set(gcbo,'value',1)
        end
        end
    end
end

for i=[gui.ChannelAddPanel gui.ChannelListPanel gui.RecControls...
        gui.TrigControls gui.TimeControls gui.ChanControls]
           try set(get(i,'children'),'enable','on'); end
end

for i=1:length(gui.pl)
    set(gui.pl,'userdata',NaN)
    Chan=get(gui.ChanControls(i),'userdata');
end

try
    for i=1:length(gui.aChanPanel)
        aChan=get(gui.aChanPanel(i),'userdata');
        set(aChan.aSourcePl,'userdata',[]);
    end
end


if get(gui.TriggerDelay,'userdata')<0
    set(gui.TriggerDelay,'userdata',0,'string','0')
elseif get(gui.TriggerDelay,'userdata')>90
    set(gui.TriggerDelay,'userdata',90,'string','0')
end
    set(gui.timetext,'position',[round(get(gui.TriggerDelay,'userdata')/10)/10 1.15 0])

set([gui.TrigMan gui.TrigChan gui.TrigLevel],'enable','off')
if isempty(get(gui.ChanList,'string')); set(gui.Trig,'value',0); 
    set(gui.Trig(1),'value',1); return; end
gui.ai.StopFcn='';
stop(gui.ai)
set(gui.HWMenu,'userdata',1)
set(gui.ai,'userdata',-1)
gui.ai.LoggingMode='Memory';
set(gui.RecStartStop,'value',0,'backgroundcolor','r','string','Record Start')
gui.ai.TriggerConditionValue=str2double(get(gui.TrigLevel,'string'));

delete(findobj('tag','gSS07thresh'))
updateRate=0.05;
Span=str2double(popupstr(gui.Refresh));
set(gui.ax,'xlim',[0 Span])

%Continuous Triggering
if get(gui.Trig(1),'value')==1
    gui.ai.TriggerDelay=0;
    gui.ai.TriggerType='Immediate';
    gui.ai.TriggerRepeat=0;
    gui.ai.TriggerFcn=@ClearPlot;
    gui.ai.samplesacquiredfcncount=floor(gui.ai.samplerate*updateRate);
    gui.ai.samplespertrigger=inf;
    set(gui.fig,'userdata',gui)
    start(gui.ai)
end

%Manual Button Press Triggering
if get(gui.Trig(2),'value')==1
    set(gui.TrigMan,'enable','on')
    gui.ai.TriggerType='Manual';
    gui.ai.TriggerRepeat=inf;
    gui.ai.TriggerFcn=@ClearPlot;
    Span=get(gui.Refresh,'string');
    Span=str2num(Span{get(gui.Refresh,'value')});
    gui.ai.TriggerDelay=-.01*get(gui.TriggerDelay,'userdata')*Span;
    gui.ai.samplesacquiredfcncount=floor(gui.ai.samplerate*updateRate);
    gui.ai.samplespertrigger=floor(Span*gui.ai.samplerate);
    start(gui.ai)
end

%Level based Software Triggering
if get(gui.Trig(3),'value')==1
    set(gui.TrigChan,'string',get(gui.ChanList,'string'),'enable','on',...
        'callback',@ChangeTrig)
    set(gui.TrigLevel,'enable','on','callback',@ChangeTrig);
    gui.ai.TriggerType='Software';
    ChanName=get(gui.TrigChan,'string');
    ChanName=ChanName(get(gui.TrigChan,'value'),:);
    %Display threshold Trace
    ChanInd=get(gui.TrigChan,'value');
    gTemp=get(gui.ax(ChanInd),'xlim');
    gui.thresh=line(gTemp,[1 1]*get(gui.TrigLevel,'userdata'),'color','k',...
       'linewidth',1,'parent',gui.ax(ChanInd),'tag',...
       'gSS07thresh','buttondownfcn',{@MoveThresh,ChanName});
    
    Span=get(gui.Refresh,'string');
    Span=str2double(Span{get(gui.Refresh,'value')});
    gui.ai.samplesacquiredfcncount=floor(gui.ai.samplerate*updateRate);
    gui.ai.TriggerDelay=-.01*get(gui.TriggerDelay,'userdata')*Span;
    gui.ai.samplespertrigger=floor(gui.ai.samplerate*Span);
   
    gui.ai.TriggerChannel=gui.ai.channel(ChanInd);
    gui.ai.TriggerCondition='Rising';
    gui.ai.TriggerConditionValue=str2double(get(gui.TrigLevel,'string'));
    gui.ai.TriggerRepeat=inf;
    gui.ai.TriggerFcn=@ClearPlot;
    start(gui.ai)
end

%Hardware Based digital triggering (throws error on winsound)
if get(gui.Trig(4),'value')==1
    gui.ai.TriggerDelay=0;
    try
        gui.ai.TriggerType='HwDigital';
    catch
        set(gui.Trig,'value',0)
        set(gui.Trig(1),'value',1)
        ChangeTrig
        return
    end
    gui.ai.TriggerRepeat=10000;
    Span=get(gui.Refresh,'string');
    Span=str2double(Span{get(gui.Refresh,'value')});
    gui.ai.TriggerFcn=@ClearPlot;
    gui.ai.samplesacquiredfcncount=floor(gui.ai.samplerate*updateRate);
    gui.ai.samplespertrigger=floor(Span*gui.ai.samplerate);
    gui.ai.TriggerConditionValue=2.5;
    start(gui.ai)
end

%With Stimulus (software) based triggering
if get(gui.Trig(5),'value')==1
    gui.ai.TriggerType='Manual';
    gui.ai.TriggerRepeat=inf;
    gui.ai.TriggerFcn=@ClearPlot;
    Span=get(gui.Refresh,'string');
    Span=str2num(Span{get(gui.Refresh,'value')});
    gui.ai.TriggerDelay=-.01*get(gui.TriggerDelay,'userdata')*Span;
    gui.ai.samplesacquiredfcncount=floor(gui.ai.samplerate*updateRate);
    gui.ai.samplespertrigger=floor(Span*gui.ai.samplerate);
    start(gui.ai)
end

scaleAx

%Clear the screen on trigger
function ClearPlot(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
set(gui.pl,'userdata',[])
set(gui.RecDuration,'userdata',0)

% GUI box to select a color for the trace
function tracecolor(obj,event)
gui=get(gcbf,'userdata');
c=uisetcolor('Select a Trace Color');
if length(c)>1
    set(gcbo,'backgroundcolor',c)
    if mean(c)>0.5; set(gcbo,'foregroundcolor','k'); 
    else set(gcbo,'foregroundcolor','w'); end
end
 
% find a file for recording to
function recbrowse(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

[fname, pname] = uiputfile([datestr(now,30),'.daq'], 'Save Data Filename',...
       [get(gui.DataMenu(1),'userdata'),datestr(now,30),'.daq']);

if isequal(fname,0) | isequal(pname,0)
   return
else
   set(gui.RecFile,'string',fullfile(pname, fname));
   set(gui.RecDir,'tag',pname);
end

set(gui.DataMenu(1),'userdata',pname);


function CleanChanLists(varargin)
%Keep channel lists and correlation traces up to date as the user adds or removes them from the
%scope.  This will kill correlation sources connected to dead interfaces and remove entire channel
%analysis panels that receive from a removed channel.

gui=get(findobj('tag','gSS07'),'userdata');
if get(gui.DataAnalysisMode,'value')==0; return; end

%Check Available Raw Channels
gTemp=findobj('Label','Select Channel','parent',gui.aFig);
delete(get(gTemp,'children'))
s=get(gui.ChanList,'string');
for i=1:length(s)
    uimenu('parent',gTemp,'Label',s{i},'callback',@AnalAddElement)
end

%Check Active Panels
killind=[];
for i=1:length(gui.aChanPanel)
    aChan=get(gui.aChanPanel(i),'userdata');
    available=0;
    for j=1:length(s)
        if strcmpi(aChan.name,s{j})
            available=1;
        end
    end
    if available==0
        %kill Analysis Channel
        killind=[killind,aChan.aChanClose];
    end
end
for i=1:length(killind)
    aCloseElement(killind(i),'')        
end

gui=get(findobj('tag','gSS07'),'userdata');

%Check All correlation sources
for i=1:length(gui.aChanPanel)
    aChan=get(gui.aChanPanel(i),'userdata');
    killind=[];
    for j=1:length(aChan.aCorrTx)
       info=get(aChan.aCorrTx(j),'userdata');
       available=0;
       switch info.type
           case 0 %Raw Trace Source
               for k=1:length(s)
                   if strcmpi(info.ChanName,s{k})
                       available=1;
                   end
               end
               if available==0
                   killind=[killind,j];
               end
           case 1
               for k=1:length(gui.aChanPanel)
                   if strcmpi(get(aChan.aCorrTx(j),'string'),get(gui.aChanPanel(k),'title'))
                       avialable=1;
                   end
               end
               if available==0
                   killind=[killind,j];
               end
       end
    end
    delete(aChan.aCorrPl(killind))
    delete(aChan.aCorrTx(killind))
    aChan.aCorrPl(killind)=[];
    aChan.aCorrTx(killind)=[];
    
    delete(get(aChan.aCorrAdd,'children'))
    aChan.aCorrChan=[];
    for j=1:length(gui.aChanPanel)
        oaChan=get(gui.aChanPanel(j),'userdata');
        aChan.aCorrChan(j)=uimenu('parent',aChan.aCorrAdd,'label',get(gui.aChanPanel(j),'title'),...
            'checked','off','callback',{@aCorrSetup,gui.aChanPanel(i),j,1,oaChan.name});
    end
    for j=1:length(s)
        aChan.aCorrChan(j+length(gui.aChanPanel))=uimenu('parent',aChan.aCorrAdd,...
        'label',[s{j},' Raw'],'checked','off',...
        'callback',{@aCorrSetup,gui.aChanPanel(i),j,0,s{j}});
        if j==1; set(aChan.aCorrChan(j+length(gui.aChanPanel)),'separator','on'); end
    end
    %Re-check the active traces in the newly created context menus
    for j=1:length(aChan.aCorrChan)
        for k=1:length(aChan.aCorrTx)
            if strcmpi(get(aChan.aCorrTx(k),'string'),get(aChan.aCorrChan(j),'label'))
                set(aChan.aCorrChan(j),'checked','on')
            end
        end
    end
    
    set(gui.aChanPanel(i),'userdata',aChan)
end

%update correlation context menu

%Initialize and Start the AI
function initAI(varargin)
gui=get(findobj('tag','gSS07'),'userdata');

%Error Checking in GUI boxes
if gcbo==gui.SRate
    sRate=str2double(get(gui.SRate,'string'));
    if isnan(sRate)
        sRate=get(gui.SRate,'userdata');
        set(gui.SRate,'string',num2str(sRate));
    else
        set(gui.SRate,'userdata',sRate)
    end
    if sRate<100
        sRate=100;
        set(gui.SRate,'userdata',100,'string','100')
    end

elseif gcbo==gui.TriggerDelay
    tDelay=str2double(get(gui.TriggerDelay,'string'));
    if isnan(tDelay)
        tDelay=get(gui.TriggerDelay,'userdata');
        set(gui.TriggerDelay,'string',num2str(tDelay));
    else
        set(gui.TriggerDelay,'userdata',tDelay)
    end
end

temp=gui.ai.StopFcn;
gui.ai.StopFcn='';
stop(gui.ai)

tempsrate=str2double(get(gui.SRate,'string'));
stuff=daqhwinfo(gui.ai);
if tempsrate>stuff.MaxSampleRate; tempsrate=stuff.MaxSampleRate; end
if tempsrate<stuff.MinSampleRate; tempsrate=stuff.MinSampleRate; end
gui.ai.samplerate=tempsrate;
set(gui.SRate,'string',num2str(gui.ai.samplerate),'userdata',gui.ai.samplerate);

try stop(gui.sound); gui.sound.samplerate=gui.ai.samplerate; end

Span=get(gui.Refresh,'string');
Span=str2double(Span(get(gui.Refresh,'value'),:));
set(gui.Refresh,'userdata',Span)
set(gui.ax,'xlim',[0 Span])

gui.ai.samplesacquiredfcncount=...
    round(gui.ai.samplerate*0.05);
gui.ai.samplesacquiredfcn={@SAF,gui.fig,gui.ai.samplerate,round(gui.ai.samplerate*0.05)};
gui.ai.samplespertrigger=inf;
gui.ai.StopFcn=temp;
gui.ai.LogToDiskMode='Index';
try start(gui.ai); catch return; end

listens=get(findobj('string','Listen'),'value');
if iscell(listens); listens=cell2mat(listens); end
if sum(listens)>0
    AudioFeedback(findobj('string','Listen','Value',1),'');
end

ChangeTrig

%Control Recording
function GoStartStop(varargin)
gui=get(findobj('tag','gSS07'),'userdata');
if isempty(gui.ChanControls); 
    set(gui.RecStartStop,'value',0,'string','Record Start','backgroundcolor','r');...
        return;
end

switch get(gui.RecStartStop,'value')
    case 0
        for i=[gui.ChannelAddPanel gui.ChannelListPanel gui.RecControls...
                gui.TrigControls gui.TimeControls gui.ChanControls]
            set(get(i,'children'),'enable','on')
        end        
        set(gui.RecStartStop,'backgroundcolor','r','string','Record Start')
        stop(gui.ai)
        ChangeTrig
    case 1
        for i=[gui.ChannelAddPanel gui.ChannelListPanel gui.RecControls...
                gui.TrigControls gui.TimeControls]
            set(get(i,'children'),'enable','off')
        end
        if get(gui.Trig(2),'value')==1; set(gui.TrigMan,'enable','on'); end
        set(gui.RecDir,'enable','on')
        set([gui.ChanList gui.ChanAudio],'enable','on')
        set(gui.RecStartStop,'enable','on')
        set(gui.RecStartStop,'Backgroundcolor','g','string','Record Stop')
        stop(gui.ai)
        if gui.ai.samplesavailable>0
            data=getdata(gui.ai,gui.ai.samplesavailable);
        end
        gui.ai.LoggingMode='Disk&Memory';
        
        fullstring=get(gui.RecFile,'string');
        slashes=strfind(fullstring,'\');
        pname=fullstring(1:slashes(end));
        fname=fullstring((slashes(end)+1):end);
        %Itterate Files (other than temp.daq), matlab's analoginput log
        %type doesn't seem to automatically index even when in index mode
        if strcmpi(fname,'temp.daq')
            gui.ai.LogFileName=get(gui.RecFile,'string');
        else
            fname=fname(1:end-4);
            tempfname=fname;
            gTemp=dir(pname);
            j=1;
            while 1==1
                p=0;
                for i=1:length(gTemp)
                    if strcmpi([tempfname,'.daq'],gTemp(i).name)
                        tempfname=[fname,num2str(j)];
                        j=j+1;
                        p=1;
                    end
                end
                if p==0
                    break
                end
            end
            gui.ai.LogFileName=[pname,tempfname,'.daq'];
            get(gui.ai,'LogFileName')
        end

        if get(gui.RecMeta,'value')==1
            %Store MetaData in an associated file
            mfname=gui.ai.LogFileName;
            mfname=[mfname(1:(end-4)),'_info.txt'];
            s=get(gui.ExperimentText,'string');
            try delete(mfname); end
            a=fopen(mfname,'w');
            siz=size(s);
            fwrite(a,datestr(now));
            fprintf(a,'\n');
            for i=1:siz(1)
                fwrite(a,s(i,:));
                fprintf(a,'\n');
            end
            fclose(a);
        end
        
        %Check for DIO Log Sync
        if get(gui.DIOLogLink,'value')&(get(gui.DIOLog,'value')==0)
            set(gui.DIOLog,'value',1)
            DIOLogRun(gui.DIOLog,'RecSync');
        end
        
        switch get(gui.Rec(1),'value')
            case 0  %Indefinite
                %gui.ai.samplespertrigger=inf;
            case 1  %Fixed Duration
                duration=str2double(get(gui.RecDuration,'string'));
                updateRate=0.05;%get(gui.UpdateRate,'userdata');
                duration=ceil(duration/updateRate)*updateRate;
                set(gui.RecDuration,'string',num2str(duration),'userdata',duration)
                
                gui.ai.samplespertrigger=floor(duration*gui.ai.samplerate/...
                    gui.ai.samplesacquiredfcncount)*gui.ai.SamplesAcquiredFcnCount;
                gui.ai.StopFcn='spikeHound(''ChangeTrig'')';
        end
        set(gui.HWMenu,'userdata',1)
        gui.OverFlag=0;
        set(gui.ai,'userdata',-1)
        set(gui.fig,'userdata',gui)
%         gui.ai.TriggerRepeat=inf;
        if get(gui.Trig(1),'value'); gui.ai.TriggerRepeat=0; end
        start(gui.ai)
        
end

%Load and arbitrary stimulus file (daq,txt,mat,wav)
function LoadAStim(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
filterspec={'*.txt','(*.txt) Preformatted Pulse String (See Help)'};
[fname, pname, filterindex] = uigetfile(filterspec,'Load Pulse Train Script File',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
set(gui.DataMenu(1),'userdata',pname)

trigpulses=0;
switch filterindex
    case 1
        fid=fopen([pname,fname]);
        fline=1;

        stimshape=[];

        while 1
            fline=fgetl(fid);
            if ~ischar(fline); break; end;
            if isempty(fline); continue; end;
            if fline(1)=='#'; continue; end;
            stimshape(end+1,:)=str2num(fline);
        end
        fclose(fid);
        if isempty(stimshape); return; end

        extents=stimshape(:,2)+stimshape(:,3);
        fs=str2double(get(gui.StimSRate,'string'));
        sig=zeros(1,fs*max(extents));
        time=linspace(0,max(extents),length(sig));
        s=size(stimshape);
        for i=1:s(1)
            sig((time>stimshape(i,2))&(time<extents(i)))=stimshape(i,1);
            if stimshape(i,4)
                trigpulses=[trigpulses,stimshape(i,2)];
            end
        end
end
sig(end)=0;
time=linspace(0,length(sig)/gui.ao.samplerate,length(sig));

dsig=[0,diff(sig)];
critPoints=find(dsig~=0);
Points(1:3:(length(critPoints)*3))=critPoints-1;
Points(2:3:(length(critPoints)*3))=critPoints;
Points(3:3:(length(critPoints)*3))=critPoints+1;
if Points(1)~=1; Points=[1,Points]; end
if Points(end)~=length(sig); Points=[Points,length(sig)]; end
Points(Points>length(sig))=[];
Points(Points<1)=[];
time=time(Points);
sig=sig(Points);

set(gui.AdvancedMenu(5),'userdata',trigpulses)
set(gui.StimSRate,'string',num2str(fs));
set(gui.StimPl,'ydata',sig,'xdata',time,'visible','on')
set(get(gui.StimPl,'parent'),'xlim',[0 max(time)],'ylimmode','auto')


%Handle Bad Values in Stimulus parameters Build Stimulus and Prep AO
function StimParam
gui=get(findobj('tag','gSS07'),'userdata');
try if strcmpi(gui.ao.running,'on'); aoactive=1; else aoactive=0; end
catch return; end

stop(gui.ao)

daqinfo=daqhwinfo(gui.ao);
if get(gui.StimSRate,'userdata')<daqinfo.MinSampleRate
    gui.ao.samplerate=daqinfo.MinSampleRate;
elseif get(gui.StimSRate,'userdata')>daqinfo.MaxSampleRate
    gui.ao.samplerate=daqinfo.MaxSampleRate;
else
    gui.ao.samplerate=get(gui.StimSRate,'userdata');
end
set(gui.StimSRate,'string',num2str(gui.ao.samplerate),'userdata',gui.ao.samplerate)
    

if gcbo==gui.StimSRate
    FCNChangeSrate(gui.StimSRate,'',aoactive);
end

if get(gui.StimMode(1),'value')==1
    set(gui.StimModeTrig,'enable','on')
    set(gui.StimModeStart,'enable','off','value',0,'string','Start')
else
    set(gui.StimModeTrig,'enable','off')
    set(gui.StimModeStart,'enable','on','value',0,'string','Start')
end

vals=[gui.StimSRate,gui.StimShape(1:4),gui.StimTetDur,gui.StimTetInt,gui.StimRepeat,...
    gui.StimAmp]; 

if get(gui.StimType(4),'value')==1 %Load Arbitrary Stimulus
    %disable Signal Gen Interface
    set(vals,'enable','off')
    set(gui.StimTypeLoad,'enable','on')
    AllTraces=[gui.StimPlPulseDur gui.StimPlDelay1 gui.StimPlDelay2 gui.StimPlDelay3,...
        gui.StimPlTetDur gui.StimPlTetInt];
    set(AllTraces,'visible','off');
    return
else
    set(vals,'enable','on')
    set(gui.StimTypeLoad,'enable','off')
end

    set(gui.StimShape,'enable','off')
    set(gui.StimTetDur,'enable','off')
    set(gui.StimTetInt,'enable','off')
    
if get(gui.StimType(1),'value')==1
    set(gui.StimShape(1:2),'enable','on')
elseif get(gui.StimType(2),'value')==1
    set(gui.StimShape(1:3),'enable','on')
elseif get(gui.StimType(3),'value')==1
    set(gui.StimShape,'enable','on')
    set(gui.StimTetDur,'enable','on')
    set(gui.StimTetInt,'enable','on')
end

    set(gui.StimRepeat,'enable','off')
if get(gui.StimMode(2),'value')==1
    set(gui.StimRepeat,'enable','on')
end

set(gui.StimViolation,'visible','off')

%Check Parameters %Sample Rate %Delays
%StimShape(1:4) = dur +3x delay
%StimTetDur & StimTetInt = Tetanic Params
%StimRepeat = Total Signal Duration
PulseDur=get(gui.StimShape(1),'userdata');
Delay1=get(gui.StimShape(2),'userdata');
Delay2=get(gui.StimShape(3),'userdata');
Delay3=get(gui.StimShape(4),'userdata');
TetDur=get(gui.StimTetDur,'userdata');
TetInt=get(gui.StimTetInt,'userdata');
Amp=get(gui.StimAmp,'userdata');
Repeat=get(gui.StimRepeat,'userdata');

AllVals=[PulseDur, Delay1, Delay2, Delay3, TetDur, TetInt, Repeat];
violation=0;

if sum(AllVals([1,3:end])<=(1/gui.ao.samplerate))>0; violation=1; end

if PulseDur>Delay2&&get(gui.StimType(1),'value')==0; violation=1; end
if PulseDur>Delay3&&get(gui.StimType(3),'value')==1;  violation=1; end
if PulseDur>TetInt&&get(gui.StimType(3),'value')==1;  violation=1; end
if (Delay1+PulseDur)>Repeat&&get(gui.StimType(1),'value')==1&&...
        get(gui.StimMode(2),'value')==1; violation=1; end
if (Delay1+Delay2+PulseDur)>Repeat&&get(gui.StimType(2),'value')==1&&...
        get(gui.StimMode(2),'value')==1; violation=1; end
if (Delay1+Delay2+TetDur+Delay3+PulseDur)>Repeat&&get(gui.StimType(3),'value')==1&&...
        get(gui.StimMode(2),'value')==1; violation=1; end
if TetInt>TetDur&&get(gui.StimType(3),'value')==1;  violation=1; end

if Amp>10|isempty(Amp); set(gui.StimAmp,'string','10','userdata',10); Amp=10; end
if Amp<=0; set(gui.StimAmp,'string','0','userdata',0); Amp=0; end

if violation==1
    set(gui.StimViolation,'visible','on','string','Stimulus Parameter Error',...
        'fontsize',.7)
    return
end

%Construct Signal

if get(gui.StimType(1),'value')==1
        sig=zeros(1,round(gui.ao.samplerate*(Delay1+PulseDur)));
elseif get(gui.StimType(2),'value')==1
        sig=zeros(1,round(gui.ao.samplerate*(Delay1+Delay2+PulseDur)));
elseif get(gui.StimType(3),'value')==1
        sig=zeros(1,round(gui.ao.samplerate*(Delay1+Delay3+TetDur+PulseDur)));
end

PulseDur=round(PulseDur*gui.ao.samplerate);
Delay1=round(Delay1*gui.ao.samplerate)+1;
Delay2=round(Delay2*gui.ao.samplerate);
Delay3=round(Delay3*gui.ao.samplerate);
TetDur=round(TetDur*gui.ao.samplerate);
TetInt=round(TetInt*gui.ao.samplerate);


if get(gui.StimType(1),'value')==1 %Create Single Pulse Signal
    sig(Delay1:(Delay1+PulseDur))=Amp;
end

if get(gui.StimType(2),'value')==1 %Create Double Pulse Signal

    sig(Delay1:(Delay1+PulseDur))=Amp;
    sig((Delay1+PulseDur):(Delay1+PulseDur+Delay2))=0;
    sig((Delay1+Delay2):(Delay1+Delay2+PulseDur))=Amp;
end

if get(gui.StimType(3),'value')==1 %Create Double + Tetanic Pulse Signal
    sig(Delay1:(Delay1+PulseDur))=Amp;
    sig((Delay1+PulseDur):(Delay1+Delay2))=0;
    
    tetsig=zeros(1,TetInt);
    tetsig(1:PulseDur)=Amp;
    tetsig=repmat(tetsig,[1,floor(TetDur/TetInt)]);
    sig((Delay1+Delay2):(Delay1+Delay2+length(tetsig)-1))=tetsig;
    finalval=max(find(sig~=0));
    sig(finalval:(finalval+Delay3))=0;
    sig((finalval+Delay3):(finalval+Delay3+PulseDur))=Amp;
    finalval2=max(find(sig~=0));
    sig=sig(1:finalval2);
end

Repeat=get(gui.StimRepeat,'userdata');
if get(gui.StimMode(2),'value')==1
    sig=[sig,zeros(1,round(Repeat*gui.ao.SampleRate-length(sig)))];
end


%Update Display and Parameter Bars
sig(end)=0;
time=linspace(0,length(sig)/gui.ao.samplerate,length(sig));

dsig=[0,diff(sig)];
critPoints=find(dsig~=0);
Points(1:3:(length(critPoints)*3))=critPoints-1;
Points(2:3:(length(critPoints)*3))=critPoints;
Points(3:3:(length(critPoints)*3))=critPoints+1;
if Points(1)~=1; Points=[1,Points]; end
if Points(end)~=length(sig); Points=[Points,length(sig)]; end

Points(Points>length(sig))=[];
Points(Points<1)=[];

time=time(Points);
sig=sig(Points);

set(gui.StimPl,'xdata',time,'ydata',sig,'visible','on')

set(gui.StimAx,'xlim',[0 max(time)],'ylimmode','auto')
if get(gui.StimMode(2),'value')==1
    set(gui.StimAx,'xlim',[0 Repeat])
end

PulseDur=get(gui.StimShape(1),'userdata');
Delay1=get(gui.StimShape(2),'userdata');
Delay2=get(gui.StimShape(3),'userdata');
Delay3=get(gui.StimShape(4),'userdata');
TetDur=get(gui.StimTetDur,'userdata');
TetInt=get(gui.StimTetInt,'userdata');

AllTraces=[gui.StimPlPulseDur gui.StimPlDelay1 gui.StimPlDelay2 gui.StimPlDelay3...
    gui.StimPlTetDur gui.StimPlTetInt];
set(AllTraces,'visible','off');

set(gui.StimPlDelay1,'ydata',get(gui.StimPlDelay1,'userdata')*Amp*[1 1])
set(gui.StimPlPulseDur,'ydata',get(gui.StimPlPulseDur,'userdata')*Amp*[1 1])
set(gui.StimPlDelay2,'ydata',get(gui.StimPlDelay2,'userdata')*Amp*[1 1])
set(gui.StimPlTetInt,'ydata',get(gui.StimPlTetInt,'userdata')*Amp*[1 1])
set(gui.StimPlTetDur,'ydata',get(gui.StimPlTetDur,'userdata')*Amp*[1 1])
set(gui.StimPlDelay3,'ydata',get(gui.StimPlDelay3,'userdata')*Amp*[1 1])

if get(gui.StimType(1),'value')==1 %Single Pulse Signal
    set(gui.StimPlDelay1,'xdata',[0 Delay1],'visible','on')
    set(gui.StimPlPulseDur,'xdata',[Delay1 Delay1+PulseDur],'visible','on')
end

if get(gui.StimType(2),'value')==1 %Double Pulse Signal
    set(gui.StimPlDelay1,'xdata',[0 Delay1],'visible','on')
    set(gui.StimPlPulseDur,'xdata',[Delay1 Delay1+PulseDur],'visible','on')
    set(gui.StimPlDelay2,'xdata',[Delay1 Delay1+Delay2],'visible','on')
end

if get(gui.StimType(3),'value')==1 %Tetanic Pulse Signal
    set(gui.StimPlDelay1,'xdata',[0 Delay1],'visible','on')
    set(gui.StimPlPulseDur,'xdata',[Delay1 Delay1+PulseDur],'visible','on')
    set(gui.StimPlDelay2,'xdata',[Delay1 Delay1+Delay2],'visible','on')
    set(gui.StimPlTetInt,'xdata',[Delay1+Delay2 Delay1+Delay2+TetInt],'visible','on')
    set(gui.StimPlTetDur,'xdata',[Delay1+Delay2 finalval/gui.ao.samplerate],...
        'visible','on')
    set(gui.StimPlDelay3,'xdata',...
        [finalval/gui.ao.samplerate finalval/gui.ao.samplerate+Delay3],'visible','on')
end



%Just Load the precalculated signal on the AO
function StimLoad(varargin)
gui=get(findobj('tag','gSS07'),'userdata');

stop(gui.ao)
delete(gui.ao.channel)
gTemp=daqhwinfo(gui.ao);
addchannel(gui.ao,gTemp.ChannelIDs(1:2));
s=get(gui.OutputRange,'string');
gui.ao.channel.outputrange=str2num(s{get(gui.OutputRange,'value')});
 
if get(gui.StimSRate,'userdata')<gTemp.MinSampleRate
    gui.ao.samplerate=gTemp.MinSampleRate;
elseif get(gui.StimSRate,'userdata')>gTemp.MaxSampleRate
    gui.ao.samplerate=gTemp.MaxSampleRate/2;
else
    gui.ao.samplerate=get(gui.StimSRate,'userdata');
end
set(gui.StimSRate,'string',num2str(gui.ao.samplerate),'userdata',gui.ao.samplerate)

sig=get(gui.StimPl,'ydata');
time=get(gui.StimPl,'xdata');

rawt=linspace(0,time(end),gui.ao.samplerate*time(end));
raws=zeros(length(rawt),1);
for i=2:length(sig)
    raws((rawt>=time(i-1))&rawt<=time(i))=sig(i);
end
sig=raws;
sig=sig(:);
sig(:,2)=zeros(length(sig),1);
sig(1:round(gui.ao.samplerate*.005),2)=5;

sig(end,:)=0;

%Load Signal
gui.ao.SamplesOutputFcn='';
putdata(gui.ao,sig)

%Set Trigger Repeat Based on Stimulus Mode (single/continuous)
switch get(gui.StimMode(1),'value')
    case 0
        gui.ao.repeatoutput=inf;
    case 1
        gui.ao.repeatoutput=0;
end

if get(gui.Trig(5),'value')==1
    try trigger(gui.ai); end
end

if get(gui.StimModeHWCheck,'value')==1
    try
    set(gui.ao,'triggertype','HwDigital','stopfcn','spikeHound(''StimLoad'')')
    try 
        s=get(gui.OutputHWTS,'string');
        set(gui.ao,'HwDigitalTriggerSource',s{get(gui.OutputHWTS,'value')}); end
    start(gui.ao)
    catch
        set(gui.StimModeHWCheck,'value',0)
        StimLoad
        return
    end
else 
    set(gui.ao,'triggertype','Immediate')
end

if gcbo==gui.StimModeTrig
     start(get(gcbo,'userdata'))
end


%Update Graphics with new data
function SAF(obj,event,fig,sRate,SAFC)
gui=get(fig,'userdata');
if isempty(gui); return; end
ai=obj;
stream=strcmpi(get(gui.AdvancedMenu(7),'checked'),'on');

%Super Fast getdata call 
try
    uddobj=daqgetfield(ai,'uddobject');
    [data t abstime]=getdata(uddobj,SAFC,'double'); 
catch
    return
end
% [data t abstime]=getdata(ai,SAFC); %Slow getdata call

Span=get(gui.Refresh,'userdata');

%Display Data
if get(gui.MainFigPause,'value')==1
    set(gui.pl,'userdata',[]);
else
    
for i=1:length(gui.pl)
    try %if it happened to catch a sweep while chan was being removed (rare)
        Chan=get(gui.ChanControls(i),'userdata');
    catch
        continue
    end
    
    oldData=get(gui.pl(i),'userdata');
    extGain=get(Chan.extG,'userdata');
    if length(oldData)==1; oldData=[]; end

    if get(Chan.Show,'value')==0
        continue
    end
    plotdata=data(:,i)/extGain;
    targetLength=floor((Span*sRate)/SAFC)*SAFC;
    
    if Span*sRate>2000; downsample=1;
    else downsample=0;
    end
    
    if get(gui.HWMenu,'userdata')==1; oldData=[]; end

    %Create Display Data based on Display Mode
    if stream
        if (length(oldData)<targetLength)
            streaming=0;
            newData=[oldData(:); plotdata(:)];
        else
            streaming=1;   
            if length(oldData)==(targetLength)&&strcmpi(get(gui.AdvancedMenu(7),'checked'),'on')
                newData=oldData;
                newData(1:(end-SAFC))=newData((SAFC+1):end);
                newData((end-SAFC+1):end)=plotdata;
            else
                newData=plotdata(:);
            end
        end
    else
        streaming=0;
        if length(oldData)<targetLength
            newData=[oldData(:); plotdata(:)];
        else
            newData=plotdata(:);
        end
    end
    plotdata=newData;

    if get(Chan.ACCouple,'value')
        plotdata=plotdata-mean(plotdata);
    end
    
    if streaming==0|length(oldData)==0
        time=(0:(length(plotdata)-1))/sRate;
        set(gui.pl(i),'ydata',plotdata,'xdata',time,'userdata',newData)
    else
        set(gui.pl(i),'ydata',plotdata,'userdata',newData)
    end

end

end %Pause check end

%Output Audio to Sound Card
if get(gui.ChanAudio,'value')==1
    SelInd=get(gui.ChanList,'value');
    Chan=get(gui.ChanControls(SelInd),'userdata');
    Gain=get(Chan.VpD,'userdata');
    Gain=0.2/Gain(get(Chan.VpD,'value'));
    extGain=get(Chan.extG,'userdata');
    ylim=get(gui.ax(SelInd),'ylim');
    offset=get(Chan.Offset,'userdata');
    sig=(data(:,SelInd)-mean(data(:,SelInd))+offset*.01/Gain)*Gain/extGain;
    
    OutputUserAudio(sig,gui,sRate);
end


%Update Elapsed Recording Time
if get(gui.RecStartStop,'value')==1
    set(gui.RecStartStop,'string',...
        ['Record Stop (',num2str(.1*round(10*ai.SamplesAcquired/ai.SampleRate)),')']);
end

%If analysis window is open
if get(gui.DataAnalysisMode,'value')==1
    aRTCalc(data,t,gui,sRate,event)
end

if get(gui.HWMenu,'userdata')==1; set(gui.HWMenu,'userdata',0); end

% drawnow('update')


%Generic Function for Updating audio for feedback to the user
function OutputUserAudio(data,gui,asr)

ssr=get(gui.sound,'Userdata');
if get(gui.sound,'SamplesAvailable')<(ssr*0.25)  
    if ssr==asr    %Check to see if sample rates are matched
        putdata(gui.sound,data)        
    else    %if not, interpolate to match
        %data=resample(data,ssr,asr);
        xi=linspace(1,length(data),round(ssr/20));
        data=interp1(1:length(data),data,xi,'nearest')';
        putdata(gui.sound,data)
    end   
    
end

if strcmpi(get(gui.sound,'Running'),'off')&(get(gui.sound,'SamplesAvailable')>=(get(gui.sound,'SampleRate')*0.1))
    start(gui.sound);
end



function AudioFeedback(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
if get(obj,'value')==0
    try stop(gui.sound); delete(gui.sound); end
    return;
end
set(findobj('string','Listen'),'value',0)
set(obj,'value',1)

try stop(gui.sound); delete(gui.sound); end
gui.sound=analogoutput('winsound');
gTemp=daqhwinfo(gui.sound);
addchannel(gui.sound,gTemp.ChannelIDs(1));
try gui.sound.samplerate=gui.ai.samplerate;
catch
    if gui.ai.samplerate>gTemp.MaxSampleRate
        try gui.sound.samplerate=44100;
        catch
            gui.sound.samplerate=gTemp.MaxSampleRate;
        end
    elseif gui.ai.samplerate<gTemp.MinSampleRate
        gui.sound.samplerate=gTemp.MinSampleRate;
    end
end

gui.sound.userdata=gui.sound.samplerate;
set(gui.fig,'userdata',gui)

%Handle Trace Extraction to external figures (maroon buttons)
function FigureExtraction(obj,event,ax,type,target)
if strcmpi(type,'Delete'); delete(obj); end
delete(findobj('tag','gSS07CaptureContext'));
Buttons=findobj('userdata','figureextract');

for i=1:length(Buttons)
    CallBack=get(Buttons(i),'callback');
    gTemp=uicontextmenu('tag','gSS07CaptureContext','parent',ancestor(Buttons(i),'figure'));
    set(Buttons(i),'UIContextMenu',gTemp)
    uimenu('parent',gTemp,'Label','New Figure','callback',CallBack);
end

if strcmpi(type,'Update')||strcmpi(type,'Delete') %General Call to update context menus
    ActiveFigures=findobj('tag','gSS07Capture');
    if strcmpi(type,'Delete'); ActiveFigures(ActiveFigures==obj)=[]; end
    ContextMenus=findobj('tag','gSS07CaptureContext');
    for i=1:length(ContextMenus)
        sub=get(ContextMenus(i),'children');
        CallBack=get(sub,'callback');
        for j=1:length(ActiveFigures)
            CallBack{4}=ActiveFigures(j);
            uimenu('parent',ContextMenus(i),'label',['Figure ',num2str(ActiveFigures(j))],'callback',CallBack)
        end
    end
    return
end

fig=target;
if isempty(fig)
    fig=figure('tag','gSS07Capture','deletefcn',{@FigureExtraction,[],'Delete',[]});
    a = [.20:.05:0.95];
    b(:,:,1) = repmat(a,16,1)';
    b(:,:,2) = repmat(a,16,1);
    b(:,:,3) = repmat(flipdim(a,2),16,1);
    hpt=uipushtool('CData',b,'TooltipString','Property Editor',...
        'ClickedCallback','spikeHound(''gPropEdit'')');
    a=axes;
end
figure(fig)
a=gca;

gui=get(findobj('tag','gSS07'),'userdata');
switch type
    case 'scope' %Main scope (all of the gui.pl traces)
        h=copyobj(gui.pl,a);
        s=get(gui.ChanList,'string');
        set(h,{'DisplayName'},s)            
        title(datestr(clock))
        xlabel('Time (s)')
        ylabel('Amplitude (V)')
    case 'RTaSourceAx' %Real-time Source
        pl=get(ax,'children');
        h=copyobj(pl,a);
        delete(findobj(h,'color','k'))
        for i=1:length(h)
            try set(h(i),'xdata',get(h(i),'xdata')/gui.ai.samplerate); end
        end
        title([get(get(ax,'parent'),'Title'),' - ',datestr(clock)])
        xlabel('Time (s)')
        ylabel('Amplitude (V)')
    case 'RTaCorrAx' %Real-time Correlation
        pl=findobj('parent',ax,'type','line');
        tx=findobj('parent',ax,'type','text');
        for i=1:length(tx)
            info=get(tx(i),'userdata');
            c=get(pl(i),'color');
            line((0:(length(info.raw)-1))/gui.ai.samplerate,info.raw,'parent',a,'DisplayName',info.name,'color',c)
        end
        title([get(get(ax,'parent'),'Title'),' Correlation - ',datestr(clock)])
        xlabel('Time (s)')
        ylabel('Amplitude (V)')
    case 'RTaAnalAx' %Real-time Analysis
        aChan=get(get(ax,'parent'),'userdata');
        pl=get(ax,'children');
        h=copyobj(pl,a);
        title([get(get(ax,'parent'),'Title'),' Analysis - ',datestr(clock)])
        xstr=get(aChan.aXval,'string');
        xstr=xstr{get(aChan.aXval,'value')};
        ystr=get(aChan.aYval,'string');
        ystr=ystr{get(aChan.aYval,'value')};
        xlabel(xstr)
        ylabel(ystr)
        set(h,'DisplayName',[xstr,' vs. ', ystr])
    case 'FFTAx'
        aChan=get(get(ax,'parent'),'userdata');
        pl=get(ax,'children');
        h=copyobj(pl,a);
        title([get(get(ax,'parent'),'Title'),' Fourier Spectrum - ',datestr(clock)])
    case 'SpectAx'
        aChan=get(get(ax,'parent'),'userdata');
        pl=get(ax,'children');
        h=copyobj(pl,a);
        set(a,'xlim',[min(get(h,'xdata')) max(get(h,'xdata'))],'ylim',[min(get(h,'ydata')) max(get(h,'ydata'))])
        title([get(get(ax,'parent'),'Title'),' Spectrogram - ',datestr(clock)])
        xlabel('Time (s)')
        ylabel('Frequency (kHz)')
end

FigureExtraction([],[],[],'Update',[]) %Recursively call this function to update contextmenus

% Take the current Scope screen signals into a .mat file
function CaptureScopeRaw(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
if isempty(gui.ax); return; end

[fname, pname] = uiputfile('*.mat', 'Target File Name for Scope Contents',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
set(gui.DataMenu(1),'userdata',pname)

time=get(gui.pl(1),'xdata')/gui.ai.samplerate;
time=time(:);
data=zeros(length(time),length(gui.pl));
for i=1:length(gui.pl)
    data(:,i)=get(gui.pl(i),'ydata');
end

save([pname,fname],'time','data')

% Load .daq file into a separate figure window
function LoadDaq
gui=get(findobj('tag','gSS07'),'userdata');
[fname, pname] = uigetfile('*.daq', 'Load Data File',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) | isequal(pname,0)
   return
end
set(gui.DataMenu(1),'userdata',pname)
[data time abstime ev daqinfo]=daqread([pname,fname]);
figure
plot(time,data)
set(gca,'xlim',[min(time) max(time)],'ylim',daqinfo.ObjInfo.Channel(1).InputRange)
legend(daqinfo.ObjInfo.Channel.ChannelName)

    a = [.20:.05:0.95];
    b(:,:,1) = repmat(a,16,1)';
    b(:,:,2) = repmat(a,16,1);
    b(:,:,3) = repmat(flipdim(a,2),16,1);
    hpt=uipushtool('CData',b,'TooltipString','Property Editor',...
        'ClickedCallback','spikeHound(''gPropEdit'')');

title([fname,'  -  ',daqinfo.HwInfo.DeviceName,'  -  ',datestr(abstime)])
xlabel('Time (s)')
ylabel('Amplitude (V)')

%Convert a .daq file into a .mat file for those without DAQ toolbox
function ConvertMAT
gui=get(findobj('tag','gSS07'),'userdata');
[fname, pname] = uigetfile('*.daq', 'File to be Converted (*.daq to *.mat)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) | isequal(pname,0)
   return
end

set(gui.DataMenu(1),'userdata',pname)

[fname2, pname2] = uiputfile('*.mat', 'Target MATLAB File Name (*.daq to *.mat)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname2,0) | isequal(pname2,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname2)

get(gui.DataMenu(1),'userdata')
[data time abstime events daqinfo]=daqread([pname,fname]);
save([pname2,fname2],'data','time','abstime','events','daqinfo');

%Export a .daq file into a comma delimited text file
function ConvertTXT
gui=get(findobj('tag','gSS07'),'userdata');
%dlmwrite
[fname, pname] = uigetfile('*.daq', 'File to be Converted (*.daq to *.txt)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname)
[fname2, pname2] = uiputfile('*.txt', 'Target Text File Name (*.daq to *.txt)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname2,0) || isequal(pname2,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname2)

[data time abstime ev daqinfo]=daqread([pname,fname]);
out=[time,data];
dlmwrite([pname2,fname2],out,'precision','%.12f')

%Convert a .daq recording into a wav audio file (one or 2 channels only)
function ConvertWAV
gui=get(findobj('tag','gSS07'),'userdata');
%dlmwrite
[fname, pname] = uigetfile('*.daq', 'File to be Converted (*.daq to *.wav audio)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname)
[fname2, pname2] = uiputfile('*.wav', 'Target WAV File Name (*.daq to *.wav)',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname2,0) || isequal(pname2,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname2)
[data time abstime ev daqinfo]=daqread([pname,fname],'Triggers',1);
oldFs=daqinfo.ObjInfo.SampleRate;
newFs=44100;
newTime=linspace(min(time),max(time),(max(time)-min(time))*newFs);
s=size(data); if s(2)>2; s(2)=2; end
for i=1:s(2)
    newData(:,i)=interp1(time,data(:,i),newTime,'spline');
end
newData=newData/max(max(newData));
wavwrite(newData,newFs,16,[pname2,fname2])

%load a saved .fig file (important if they don't have matlab)
function OpenFIG(varargin)
gui=get(findobj('tag','gSS07'),'userdata');
[fname, pname] = uigetfile('*.fig', 'Matlab Figure File',...
    get(gui.DataMenu(1),'userdata'));
if isequal(fname,0) || isequal(pname,0)
   return
end
    set(gui.DataMenu(1),'userdata',pname)
    open([pname,fname])

%Splash Screen with info about authors
function Aboutgprime(varargin)
gui=get(findobj('tag','gSS07'),'userdata');
tempFig=figure('position',[0 0 772 301],'menubar','none','numbertitle','off','name',...
    'Spike Hound','resize','off','tag','gSS07splash','windowstyle','modal',...
    'color',[.8 .9 .8]);
centerfig(tempFig)
axes('position',[0 0 1 1],'color',[.8 .9 .8],'yticklabel',[]...
    ,'xticklabel',[],'xlim',[0 1],'ylim',[0 1],'xcolor',[.4 .4 .4],'ycolor',[.4 .4 .4]);
box on
aTemp=imread('spikehoundSplash_Scope.jpg');
image(aTemp)

text(400,200,['v',gui.version],'fontsize',25,'fontweight','bold',...
    'horizontalalignment','left','fontname','Alba Super')
text(400,235,'lottg@janelia.hhmi.org','fontsize',12,'fontweight','bold',...
    'horizontalalignment','left')
% text(0.5, 0.6,'Physiological Recording & Identification of Multiple Events',...
%     'fontsize',10,'fontweight','bold','horizontalalignment','center')
text(400,260,'(c)2008 Gus Kaderly Lott III, PhD','fontsize',12,'fontweight','bold',...
    'horizontalalignment','left')
% text(0.5,0.25,'HHMI - Janelia Farm Research Campus','fontsize',15,'fontweight','bold',...
%     'horizontalalignment','center','color',[.7 0 0])
text(25,290,'MATLAB(R). Copyright 1984 - 2008 The MathWorks, Inc.','fontsize',10,'fontweight','bold',...
    'horizontalalignment','left','fontname','Comic Sans MS','color',[.1 .7 .1])



%RT Analysis Functionality%%
function InitAnalysis(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
if get(obj,'value')==0
    delete(findobj('tag','gSS07anal'))
    gui.aChanPanel=[];
    set(gui.fig,'userdata',gui)
    return;
end

gui.aFig=figure('position',[100 100 1200 5],'numbertitle','off','resize','on',...
    'name','Spike Hound - Live Data Analysis - Gus K. Lott III, PhD','menubar','none',...
    'tag','gSS07anal','deletefcn','set(findobj(''string'',''Live Data Analysis''),''value'',0)',...
    'resizefcn',@aFigResize);
set(gui.aFig,'renderer',get(gui.fig,'renderer'),'WVisual',get(gui.fig,'WVisual'))
set(gui.aFig,'WindowScrollWheelFcn',@AnalScaleScroll)
set(gui.aFig,'KeyPressFcn',@AnalScaleScrollKey)
set(gui.aFig,'windowbuttonupfcn',...
    'set(gcf,''windowbuttonmotionfcn'','''');')
centerfig(gui.aFig)
gui.aChanPanel=[];

gTemp=uimenu('Label','Select Channel');
s=get(gui.ChanList,'string');
for i=1:length(s)
    uimenu('parent',gTemp,'Label',s{i},'callback',@AnalAddElement)
end
set(gui.fig,'userdata',gui)

obj=findobj('parent',gTemp,'Label',s{get(gui.ChanList,'value')});
AnalAddElement(obj,'');

function aFigResize(obj,event)
p=get(obj,'position');
p(3)=1200;
set(obj,'position',p)

function AnalAddElement(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ChanName=get(obj,'Label');

%________________________________
%Data Display Code
screensize=get(0,'ScreenSize');
p=get(gui.aFig,'position');
set(gui.aChanPanel,'units','pixels');
p=[p(1) p(2) 1200 260*(length(gui.aChanPanel)+1)];
if (p(2)+p(4)+40)>screensize(4)
    p(2)=screensize(4)-p(4)-40;
end
for i=1:length(gui.aChanPanel)
    set(gui.aChanPanel(i),'position',[5 5+(i-1)*260 1190 250])
end
set(gui.aFig,'position',p)



ver=1;
for i=1:length(gui.aChanPanel)
    aChan=get(gui.aChanPanel(i),'userdata');
    if strcmpi(aChan.name,ChanName)
        ver=ver+1;
    end
end
found=0;
while found==0
    if ~isempty(findobj('title',[ChanName,' v',num2str(ver)]))
        ver=ver+1;
    else
        found=1;
    end
end

aChan=[];
gui.aChanPanel(end+1)=uipanel('units','pixels','position',[5 5+length(gui.aChanPanel)*260 1190 250],...
    'backgroundcolor',[1 1 .7],'title',[ChanName,' v',num2str(ver)]);

set(gui.aChanPanel,'units','normalized');

aChan.name=ChanName;
aChan.aChanClose=uicontrol('style','pushbutton','string','x','units','pixels',...
    'position',[1175 230 15 15],'fontweight','bold','callback',@aCloseElement,...
    'parent',gui.aChanPanel(end));
aChan.Listen=uicontrol('style','toggle','string','Listen','units','pixels',...
    'position',[1075 230 100 15],'fontweight','normal','callback',@AudioFeedback,...
    'parent',gui.aChanPanel(end),'backgroundcolor',[1 .7 .7]);
aChan.Pause=uicontrol('style','toggle','string','Pause','units','pixels',...
    'position',[975 230 100 15],'fontweight','normal',...
    'parent',gui.aChanPanel(end),'backgroundcolor',[.8 .8 .8],'callback',{@aLinkPause,gui.aChanPanel(end)});
aChan.LinkPause=uicontrol('style','Checkbox','string','','units','pixels',...
    'position',[960 230 15 15],'fontweight','normal',...
    'parent',gui.aChanPanel(end),'backgroundcolor',[.8 .8 .8]);

aChan.aSourceAx=axes('units','pixels','position',[25 50 500 165],'color',[.8 .9 .8],...
    'xlim',[0 .1],'ylim',[-1 1],'xgrid','on','ygrid','on','fontsize',8,...
    'parent',gui.aChanPanel(end),'xaxislocation','top','box','on','userdata',[]);
aChan.aSourcePl=line(nan,nan,'color','b','userdata',[]);
aChan.aSourceDrag=text(0,0,'+','fontweight','bold','fontsize',15,'horizontalalignment','center','buttondownfcn',{@aDragSource,gui.aChanPanel(end)});
aChan.aThresh1Pl=line([0 1],[1 1]*.05,'color','k','linewidth',2,'visible','off',...
    'buttondownfcn',{@aDragThresh,1,gui.aChanPanel(end)});
aChan.aThresh2Pl=line([0 1],[1 1]*.02,'color','k','linewidth',2,'visible','off',...
    'buttondownfcn',{@aDragThresh,2,gui.aChanPanel(end)});
aChan.aSourceCap=uicontrol('style','pushbutton','string','','parent',...
    gui.aChanPanel(end),'units','pixels','position',[510 200 15 15],...
    'enable','on','callback',{@FigureExtraction,aChan.aSourceAx,'RTaSourceAx',[]},'backgroundcolor',[.7 .2 .2],...
    'tag','','userdata','figureextract');

%Context Menu for Filter Options
aChan.DataContext=uicontextmenu('tag','gSS07Context');
set([aChan.aSourceAx aChan.aSourcePl aChan.aThresh1Pl aChan.aThresh2Pl],'UIContextMenu',aChan.DataContext);
    aChan.aHP=uimenu('parent',aChan.DataContext,'label',...
        'High Pass Filter','userdata',ChanName);
        aChan.aHPMenu(1)=uimenu('parent',aChan.aHP,'label','None','checked','on','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','10','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','100','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','200','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','300','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','500','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','800','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','1000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aHPMenu(end+1)=uimenu('parent',aChan.aHP,'label','5000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        set(aChan.aHPMenu,'userdata',aChan.aHPMenu)
        
    aChan.aLP=uimenu('parent',aChan.DataContext,'label',...
        'Low Pass Filter');
        aChan.aLPMenu(1)=uimenu('parent',aChan.aLP,'label','None','checked','on','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','10000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','5000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','3000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','2000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','1000','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','500','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','200','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        aChan.aLPMenu(end+1)=uimenu('parent',aChan.aLP,'label','100','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
        set(aChan.aLPMenu,'userdata',aChan.aLPMenu)
        
    aChan.aAF=uimenu('parent',aChan.DataContext,'label','Custom Filter','enable','off');
        aChan.aAFload=uimenu('parent',aChan.aAF,'label','Load Coefficients',...
            'callback','');
    aChan.a60Hz=uimenu('parent',aChan.DataContext,'label','60 Hz Notch','callback',{@aRTFiltCalc,gui.aChanPanel(end)});
    aChan.aRectify=uimenu('parent',aChan.DataContext,'label',...
        'Rectify (for detection only)','callback',...
        ['if strcmpi(get(gcbo,''checked''),''on''); set(gcbo,''checked'',''off'');',...
        ' else set(gcbo,''checked'',''on''); end']);
    aChan.aInvert=uimenu('parent',aChan.DataContext,'label',...
        'Invert Signal','callback',...
        ['if strcmpi(get(gcbo,''checked''),''on''); set(gcbo,''checked'',''off'');',...
        ' else set(gcbo,''checked'',''on''); end']);
    aChan.aFiltDisp=uimenu('parent',aChan.DataContext,'label','Display Total Filter Spectrum',...
        'separator','on','callback',{@dispFiltSpect,gui.aChanPanel(end)});
    aChan.aWindowCenter=uimenu('parent',aChan.DataContext,'label',...
        'Window Center on Threshold','separator','on');
        aChan.aWindowCent(1)=uimenu('parent',aChan.aWindowCenter,'label','0%');
        aChan.aWindowCent(2)=uimenu('parent',aChan.aWindowCenter,'label','25%');
        aChan.aWindowCent(3)=uimenu('parent',aChan.aWindowCenter,'label','50%',...
            'checked','on');
        aChan.aWindowCent(4)=uimenu('parent',aChan.aWindowCenter,'label','75%');
        set(aChan.aWindowCent,'userdata',aChan.aWindowCent,'callback',...
        'set(get(gcbo,''userdata''),''checked'',''off''); set(gcbo,''checked'',''on'')');

aChan.aSpect=uicontrol('style','toggle','string','Spectrogram','parent',...
    gui.aChanPanel(end),'units','pixels','position',[5 23 65 19],...
    'enable','on','callback',{@ActivateSpectrum,gui.aChanPanel(end)},'fontsize',7);
aChan.aFFT=uicontrol('style','toggle','string','FFT','parent',...
    gui.aChanPanel(end),'units','pixels','position',[5 4 65 18],...
    'enable','on','callback',{@ActivateSpectrum,gui.aChanPanel(end)},'fontsize',7);
    

aChan.aFFTAx=axes('units','pixels','position',[25 50 500 165],'color',[.8 .9 .8],...
    'xlimmode','auto','ylimmode','auto','xgrid','on','ygrid','on','fontsize',8,...
    'parent',gui.aChanPanel(end),'xaxislocation','top','box','on','userdata',[],'visible','off',...
    'xscale','linear','yscale','linear');
aChan.aFFTPl=line(nan,nan,'color','b','userdata',[],'visible','off','linewidth',2);
aChan.aFFTCap=uicontrol('style','pushbutton','string','','parent',...
    gui.aChanPanel(end),'units','pixels','position',[510 200 15 15],...
    'enable','on','callback',{@FigureExtraction,aChan.aFFTAx,'FFTAx',[]},'backgroundcolor',[.7 .2 .2],...
    'tag','','userdata','figureextract','visible','off');

aChan.aSpectAx=axes('units','pixels','position',[40 50 485 165],'color',[.8 .9 .8],...
    'xlimmode','auto','ylimmode','auto','xgrid','on','ygrid','on','fontsize',8,...
    'parent',gui.aChanPanel(end),'xaxislocation','top','box','on','userdata',[],'visible','off',...
    'xscale','linear','yscale','linear');
aChan.aSpectIm=image([0],'parent',aChan.aSpectAx,'visible','off');
set(aChan.aSpectAx,'visible','off','xaxislocation','top','xlimmode','auto','ylimmode','auto','ydir','normal')
ylabel('Frequency (kHz)')
aChan.aSpectCap=uicontrol('style','pushbutton','string','','parent',...
    gui.aChanPanel(end),'units','pixels','position',[510 200 15 15],...
    'enable','on','callback',{@FigureExtraction,aChan.aSpectAx,'SpectAx',[]},'backgroundcolor',[.7 .2 .2],...
    'tag','','userdata','figureextract','visible','off');
colormap(hsv(256))

uicontrol('style','text','string','Time Scale','parent',gui.aChanPanel(end),...
    'backgroundcolor', get(gui.aChanPanel(end),'backgroundcolor')-.1,'units','pixels',...
    'position',[75 2 55 40],'horizontalalignment','left');
aChan.aSourceTime=uicontrol('style','popupmenu','string',{'.05','0.1','0.2','0.4','0.8','1','2','5','10'},...
    'parent',gui.aChanPanel(end),'units','pixels','position',[76 5 53 20],'backgroundcolor','w',...
    'enable','on','callback',{@aTraceSet,gui.aChanPanel(end)},'value',2,'userdata',0.1);

uicontrol('style','text','string','Voltage Range','parent',gui.aChanPanel(end),...
    'backgroundcolor', get(gui.aChanPanel(end),'backgroundcolor')-.1,'units','pixels',...
    'position',[140 2 100 40],'horizontalalignment','left');
aChan.aSourceAmp=uicontrol('style','popupmenu','string',{' '},'parent',gui.aChanPanel(end),...
    'units','pixels','position',[141 5 98 20],'backgroundcolor','w',...
    'enable','on','callback',{@aTraceSet,gui.aChanPanel(end)},'userdata',.1);
s={'[-20 20]','[-10  10]','[-5  5]','[-2  2]','[-1  1]','[-0.5  0.5]','[-0.2  0.2]','[-0.1  0.1]',...
    '[-0.05  0.05]','[-0.01  0.01]','[-0.005  0.005]','[-0.001  0.001]','[-0.0005  0.0005]','[-0.0001  0.0001]'};
set(aChan.aSourceAmp,'string',s,'value',6)

uicontrol('style','text','string','Window (sec) ','parent',gui.aChanPanel(end),...
    'backgroundcolor', get(gui.aChanPanel(end),'backgroundcolor')-.1,'units','pixels',...
    'position',[255 2 100 40]);
aChan.aWindow=uicontrol('style','edit','string','0.01','parent',gui.aChanPanel(end),...
    'units','pixels','position',[256 5 98 20],'backgroundcolor','w',...
    'userdata',0.01,'callback',{@aWindowChange,gui.aChanPanel(end)});

aChan.aTH1Val=uicontrol('style','edit','string','0.05','parent',gui.aChanPanel(end),...
    'units','pixels','position',[370 25 75 20 ],'backgroundcolor','w',...
    'userdata',0.05,'enable','off','callback',{@analysisThreshSet,gui.aChanPanel(end)});
aChan.aThresh1=uicontrol('style','checkbox','string','Threshold 1','parent',gui.aChanPanel(end),...
    'units','pixels','position',[450 25 100 20],'userdata',[],'backgroundcolor',...
    get(gui.aChanPanel(end),'backgroundcolor'),'callback',{@analysisThreshSet,gui.aChanPanel(end)});
aChan.aTH2Val=uicontrol('style','edit','string','0.02','parent',gui.aChanPanel(end),...
    'units','pixels','position',[370 4 75 20],'backgroundcolor','w',...
    'userdata',0.02,'enable','off','callback',{@analysisThreshSet,gui.aChanPanel(end)});
aChan.aThresh2=uicontrol('style','checkbox','string','Threshold 2','parent',gui.aChanPanel(end),...
    'units','pixels','position',[450 4 100 20],'backgroundcolor',...
    get(gui.aChanPanel(end),'backgroundcolor'),'userdata',aChan.aThresh2Pl,'callback',...
    {@analysisThreshSet,gui.aChanPanel(end)});

aChan.aRTCorrControl=uicontrol('style','toggle','string','Correlation','parent',...
    gui.aChanPanel(end),'units','pixels','position',[540 4 100 20],...
    'enable','on','callback',@sizeAnalAx);
aChan.aCorrRST=uicontrol('style','pushbutton','string','Corr. Reset','parent',...
    gui.aChanPanel(end),'units','pixels','position',[660 4 100 20],...
    'enable','off','callback',{@CorrClear,gui.aChanPanel(end)});
aChan.aRTAnalysis=uicontrol('style','toggle','string','Analysis','parent',...
    gui.aChanPanel(end),'units','pixels','position',[800 4 100 20],...
    'enable','on','callback',@sizeAnalAx);
aChan.aAnalysisRST=uicontrol('style','pushbutton','string','Analysis Reset','parent',...
    gui.aChanPanel(end),'units','pixels','position',[920 4 100 20],...
    'enable','off','callback',{@AnalysisReset,gui.aChanPanel(end)});

%-------------------------Real Time Correlation Axis------------------------
aChan.aCorrAx=axes('units','pixels','position',[540 50 220 165],'color',[.8 .9 .8],...
    'xlim',[0 1],'ylim',[-1 1],'xgrid','on','ygrid','on','tag','correlation',...
    'parent',gui.aChanPanel(end),'xaxislocation','top','yaxislocation','right',...
    'box','on','fontsize',8);
aChan.aCorrPl=[];
aChan.aCorrTx=[];
aChan.aCorrTxContext=[];

aChan.CorrContext=uicontextmenu('tag','gSS07Context');
set([aChan.aCorrAx],'UIContextMenu',aChan.CorrContext);
    aChan.aCorrAdd=uimenu('parent',aChan.CorrContext,'label',...
        'Activate Channel');
    aChan.aCorrChan=[];
    for i=1:(length(gui.aChanPanel)-1)
        oaChan=get(gui.aChanPanel(i),'userdata');
        aChan.aCorrChan(i)=uimenu('parent',aChan.aCorrAdd,'label',get(gui.aChanPanel(i),'title'),...
            'checked','off','callback',{@aCorrSetup,gui.aChanPanel(end),i,1,oaChan.name});
    end
    aChan.aCorrChan(end+1)=uimenu('parent',aChan.aCorrAdd,'label',get(gui.aChanPanel(end),'title'),...
        'checked','off','callback',{@aCorrSetup,gui.aChanPanel(end),i,1,aChan.name});
    
    s=get(gui.ChanList,'string');
    for i=1:length(s)
        aChan.aCorrChan(i+length(gui.aChanPanel))=uimenu('parent',aChan.aCorrAdd,...
        'label',[s{i},' Raw'],'checked','off',...
        'callback',{@aCorrSetup,gui.aChanPanel(end),i,0,s{i}});
        if i==1; set(aChan.aCorrChan(i+length(gui.aChanPanel)),'separator','on'); end
    end
    
    aChan.aCorrTrig(1)=uimenu('parent',aChan.CorrContext,'label','Trigger Location');
        aChan.aCorrTrig(2)=uimenu('parent',aChan.aCorrTrig(1),'label','Threshold Cross','checked','on');
        aChan.aCorrTrig(3)=uimenu('parent',aChan.aCorrTrig(1),'label','Max Amplitude');
        aChan.aCorrTrig(4)=uimenu('parent',aChan.aCorrTrig(1),'label','Min Amplitude');
        set(aChan.aCorrTrig(2:4),'userdata',aChan.aCorrTrig(2:4),'callback',...
            'set(get(gcbo,''userdata''),''checked'',''off''); set(gcbo,''checked'',''on'');')
    aChan.aCorrCenter=uimenu('parent',aChan.CorrContext,'label','Center Window on Event','callback',...
        {@CorrCenterEvent,gui.aChanPanel(end)});
    aChan.CorrCenterEvent=0;
    aChan.aHideText=uimenu('parent',aChan.CorrContext,'label','Hide Text Labels','callback',{@CorrHideLabels,gui.aChanPanel(end)});
        
%Correlation Window Width
aChan.aCWindowT=uicontrol('style','text','string','Window','parent',gui.aChanPanel(end),...
    'backgroundcolor', get(gui.aChanPanel(end),'backgroundcolor')-.1,'units','pixels',...
    'position',[540 25 100 17],'tag','correlation','horizontalalignment','left');
aChan.aCWindow=uicontrol('style','edit','string','0.05','parent',gui.aChanPanel(end),...
    'units','pixels','position',[590 25 50 20],'backgroundcolor','w',...
    'userdata',0.05,'callback',{@CorrClear,gui.aChanPanel(end)},'tag','correlation');
aChan.aCResetT=uicontrol('style','text','string','Reset (s)','parent',gui.aChanPanel(end),...
    'backgroundcolor', get(gui.aChanPanel(end),'backgroundcolor')-.1,'units','pixels',...
    'position',[660 25 100 17],'tag','correlation','horizontalalignment','left');
aChan.aCResetTime=uicontrol('style','edit','string','0','parent',gui.aChanPanel(end),...
    'units','pixels','position',[710 25 50 20],'backgroundcolor','w',...
    'userdata',0,'callback',{@CorrClear,gui.aChanPanel(end)},'tag','correlation');

%Extract
aChan.aCCapture=uicontrol('style','pushbutton','string','','parent',...
    gui.aChanPanel(end),'units','pixels','position',[745 200 15 15],...
    'enable','on','callback',{@FigureExtraction,aChan.aCorrAx,'RTaCorrAx',[]},'backgroundcolor',[.7 .2 .2],...
    'tag','correlation','userdata','figureextract');


%Context Menu Should allow for Histogram production
aChan.aAnalAx=axes('parent',gui.aChanPanel(end),'units','pixels','position',[790 50 235 165],...
    'color',[.8 .8 .8],'tag','analysis','fontsize',8,'xaxislocation','top','yaxislocation','right'); box on;
aChan.aAnalPl=line(nan,nan,'color','r','parent',aChan.aAnalAx,'linestyle','none',...
    'marker','.','markeredgecolor','r','tag','analysis','userdata',[]);
aChan.aAnalCap=uicontrol('style','pushbutton','string','','parent',...
    gui.aChanPanel(end),'units','pixels','position',[1010 200 15 15],...
    'enable','on','callback',{@FigureExtraction,aChan.aAnalAx,'RTaAnalAx',[]},'backgroundcolor',...
    [.7 .2 .2],'tag','analysis','userdata','figureextract');

%Controls for Analysis Display
aChan.aEDenThresh=uicontrol('style','checkbox','string','Energy Threshold','units',...
    'pixels','backgroundcolor',[.9 .9 .6],'horizontalalignment','center',...
    'position',[1050 200 130 20],'parent',gui.aChanPanel(end),'callback',...
    {@aEDenThresh,aChan.aAnalAx,gui.aChanPanel(end)},'value',0,'enable','on','tag','analysis');
aChan.aFreqThresh=uicontrol('style','checkbox','string','Frequency Threshold','units',...
    'pixels','backgroundcolor',[.9 .9 .6],'horizontalalignment','center',...
    'position',[1050 200 130 20],'parent',gui.aChanPanel(end),'callback',...
    {@aFreqThresh,aChan.aAnalAx,gui.aChanPanel(end)},'value',0,'enable','on','visible','off','tag','analysis');

%y-axis value select, axis scale, autoaxis, log/linear
uicontrol('style','text','string','Y-Axis Metric','units','pixels',...
    'backgroundcolor',[.8 .8 .5],'horizontalalignment','center','position',...
    [1050 175 130 18],'parent',gui.aChanPanel(end),'tag','analysis');
aChan.aYval=uicontrol('style','popupmenu','units','pixels','backgroundcolor','w',...
    'horizontalalignment','left','position',[1050 150 130 20],'parent',gui.aChanPanel(end),...
    'string',{'Event Time','Rate','Interval',...
    'Amplitude (Min)','Amplitude (Max)','Peak Frequency','Energy Density'},'value',7,...
    'callback',{@AnalysisSetting,gui.aChanPanel(end)},'tag','analysis');
aChan.aYauto=uicontrol('style','checkbox','units','pixels','backgroundcolor',...
  get(gui.aChanPanel(end),'backgroundcolor'),'position',[1050 125 65 20],'value',1,'parent',...
    gui.aChanPanel(end),'string','Auto','callback',{@AnalysisSetting,gui.aChanPanel(end)},'tag','analysis');
aChan.aYlog=uicontrol('style','checkbox','units','pixels','backgroundcolor',...
    get(gui.aChanPanel(end),'backgroundcolor'),'position',[1110 125 65 20],'parent',...
    gui.aChanPanel(end),'string','Log','callback',{@AnalysisSetting,gui.aChanPanel(end)},'tag','analysis');
aChan.Ymin=uicontrol('style','edit','units','pixels','backgroundcolor','w','position',[1050 104 65 20],'parent',gui.aChanPanel(end),...
    'string','0','userdata',0,'tag','analysis','enable','off','callback',{@AnalysisSetting,gui.aChanPanel(end)});
aChan.Ymax=uicontrol('style','edit','units','pixels','backgroundcolor','w','position',[1115 104 65 20],'parent',gui.aChanPanel(end),...
    'string','1','userdata',1,'tag','analysis','enable','off','callback',{@AnalysisSetting,gui.aChanPanel(end)});

%x-axis value select, axis scale, autoaxis, log/linear
uicontrol('style','text','string','X-Axis Metric','units','pixels',...
    'backgroundcolor',[.8 .8 .5],'horizontalalignment','center',...
    'position',[1050 80 130 18],'parent',gui.aChanPanel(end),'tag','analysis');
aChan.aXval=uicontrol('style','popupmenu','units','pixels','backgroundcolor','w',...
    'horizontalalignment','left','position',[1050 55 130 20],'parent',gui.aChanPanel(end),...
    'string',{'Event Time','Rate','Interval','Amplitude (Min)',...
    'Amplitude (Max)','Peak Frequency','Energy Density'},'callback',...
    {@AnalysisSetting,gui.aChanPanel(end)},'value',1,'tag','analysis');
aChan.aXauto=uicontrol('style','checkbox','units','pixels','backgroundcolor',...
    get(gui.aChanPanel(end),'backgroundcolor'),'position',[1050 30 65 20],...
    'parent',gui.aChanPanel(end),'string','Auto','value',1,'callback',...
    {@AnalysisSetting,gui.aChanPanel(end)},'enable','off','tag','analysis');
aChan.aXlog=uicontrol('style','checkbox','units','pixels','backgroundcolor',...
    get(gui.aChanPanel(end),'backgroundcolor'),'position',[1110 30 65 20],'parent',...
    gui.aChanPanel(end),'string','Log','callback',{@AnalysisSetting,gui.aChanPanel(end)},'tag','analysis');
aChan.aXwidthT=uicontrol('style','text','string','Width (s):','units','pixels',...
    'backgroundcolor',get(gui.aChanPanel(end),'backgroundcolor'),'horizontalalignment','left',...
    'position',[1050 3 130 20],'parent',gui.aChanPanel(end),'tag','analysis');
aChan.aXwidth=uicontrol('style','popupmenu','units','pixels','backgroundcolor','w',...
    'horizontalalignment','left','position',[1110 8 65 20],'parent',gui.aChanPanel(end),...
    'string',{'5','10','30','60','90','120','300'},'value',1,'callback',...
    {@AnalysisSetting,gui.aChanPanel(end)},'tag','analysis','userdata',5);
aChan.Xmin=uicontrol('style','edit','units','pixels','backgroundcolor','w','position',[1050 8 65 20],'parent',gui.aChanPanel(end),...
    'string','0','userdata',0,'tag','analysis','enable','off','visible','off','callback',{@AnalysisSetting,gui.aChanPanel(end)});
aChan.Xmax=uicontrol('style','edit','units','pixels','backgroundcolor','w','position',[1115 8 65 20],'parent',gui.aChanPanel(end),...
    'string','10','userdata',10,'tag','analysis','enable','off','visible','off','callback',{@AnalysisSetting,gui.aChanPanel(end)});


set(findobj('parent',gui.aChanPanel(end),'tag','analysis'),'visible','off')
set(findobj('parent',gui.aChanPanel(end),'tag','correlation'),'visible','off')

gui.aSourcePos(1,:)=[25 50 1120 165]; gui.aSourcePos(2,:)=[25 50 500 165]; gui.aSourcePos(3,:)=[25 50 500 165];
gui.aSourceCapPos(1,:)=[1130 200 15 15]; gui.aSourceCapPos(2,:)=[510 200 15 15]; gui.aSourceCapPos(3,:)=[510 200 15 15];
gui.aCorrPos(1,:)=[0 0 0 0]; gui.aCorrPos(2,:)=[540 50 605 165]; gui.aCorrPos(3,:)=[540 50 220 165];
gui.aCorrCapPos(1,:)=[0 0 0 0]; gui.aCorrCapPos(2,:)=[1130 200 15 15]; gui.aCorrCapPos(3,:)=[745 200 15 15];
gui.aAnalPos(1,:)=[0 0 0 0]; gui.aAnalPos(2,:)=[540 50 485 165]; gui.aAnalPos(3,:)=[790 50 235 165];
gui.aAnalCapPos(1,:)=[0 0 0 0]; gui.aAnalCapPos(2,:)=[1010 200 15 15]; gui.aAnalCapPos(3,:)=[1010 200 15 15];

set(aChan.aSourceAx,'position',gui.aSourcePos(1,:))
set(aChan.aSourceCap,'position',gui.aSourceCapPos(1,:))

gui.aSourcePos(:,1)=gui.aSourcePos(:,1)/1190;
gui.aSourcePos(:,2)=gui.aSourcePos(:,2)/230;
gui.aSourcePos(:,3)=gui.aSourcePos(:,3)/1190;
gui.aSourcePos(:,4)=gui.aSourcePos(:,4)/230;
gui.aSourceCapPos(:,1)=gui.aSourceCapPos(:,1)/1190;
gui.aSourceCapPos(:,2)=gui.aSourceCapPos(:,2)/230;
gui.aSourceCapPos(:,3)=gui.aSourceCapPos(:,3)/1190;
gui.aSourceCapPos(:,4)=gui.aSourceCapPos(:,4)/230;
gui.aCorrPos(:,1)=gui.aCorrPos(:,1)/1190;
gui.aCorrPos(:,2)=gui.aCorrPos(:,2)/230;
gui.aCorrPos(:,3)=gui.aCorrPos(:,3)/1190;
gui.aCorrPos(:,4)=gui.aCorrPos(:,4)/230;
gui.aCorrCapPos(:,1)=gui.aCorrCapPos(:,1)/1190;
gui.aCorrCapPos(:,2)=gui.aCorrCapPos(:,2)/230;
gui.aCorrCapPos(:,3)=gui.aCorrCapPos(:,3)/1190;
gui.aCorrCapPos(:,4)=gui.aCorrCapPos(:,4)/230;
gui.aAnalPos(:,1)=gui.aAnalPos(:,1)/1190;
gui.aAnalPos(:,2)=gui.aAnalPos(:,2)/230;
gui.aAnalPos(:,3)=gui.aAnalPos(:,3)/1190;
gui.aAnalPos(:,4)=gui.aAnalPos(:,4)/230;
gui.aAnalCapPos(:,1)=gui.aAnalCapPos(:,1)/1190;
gui.aAnalCapPos(:,2)=gui.aAnalCapPos(:,2)/230;
gui.aAnalCapPos(:,3)=gui.aAnalCapPos(:,3)/1190;
gui.aAnalCapPos(:,4)=gui.aAnalCapPos(:,4)/230;

aChan.filt.sRate=gui.ai.samplerate;
aChan.filt.HP.a=1; aChan.filt.HP.b=1;
aChan.filt.LP.a=1; aChan.filt.LP.b=1;
aChan.filt.f60.a=1; aChan.filt.f60.b=1;
aChan.filt.arb.a=1; aChan.filt.arb.b=1;
aChan.filt.BP.a=1; aChan.filt.BP.b=1;

predata=ones(round(0.2*gui.ai.samplerate),1);
set(aChan.aHP,'userdata',predata)
aChan.aCorrPl=[]; aChan.aCorrTx=[];

set(gui.fig,'userdata',gui)
set(gui.aChanPanel(end),'userdata',aChan)
aTraceSet([],'',get(gca,'parent'))
FigureExtraction([],[],[],'Update',[])
CleanChanLists

set([gui.aChanPanel(end) aChan.aSourceAx aChan.aChanClose aChan.Listen aChan.Pause aChan.aSourceCap ...
    aChan.aCorrAx aChan.aCCapture aChan.aFFTAx aChan.aFFTCap aChan.aSpectAx aChan.aSpectCap ...
    aChan.aAnalAx aChan.aAnalCap],'units','normalized')


function aDragSource(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');
set(gcf,'windowbuttonmotionfcn',{@DragSourceMotion,aChan,panel})

function DragSourceMotion(obj,event,aChan,panel)
a=get(aChan.aSourceAx,'currentpoint');
ylim=get(aChan.aSourceAx,'ylim');
if (a(1,2)>max(ylim))||(a(1,2)<min(ylim)); return; end
if (abs(a(1,2)))<(.05*diff(ylim)); return; end

TempRange=ylim-sign(a(1,2))*.05*diff(ylim);
set(aChan.aSourceAx,'ylim',TempRange)

function aLinkPause(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');
val=get(aChan.Pause,'value');

if get(aChan.LinkPause,'value')
    for i=1:length(gui.aChanPanel)
        aChan=get(gui.aChanPanel(i),'userdata');
        if get(aChan.LinkPause,'value')
            set(aChan.Pause,'value',val)
        end
    end
end

function aWindowChange(obj,event,panel)
aChan=get(panel,'userdata');
window=str2double(get(obj,'string'));
if isnan(window)|window<=0
    set(obj,'string',num2str(get(obj,'userdata')))
else
    set(obj,'userdata',window)
end
set(aChan.aSourceAx,'userdata',[])

function aCloseElement(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
panel=get(obj,'parent');
aChan=get(panel,'userdata');
delete(aChan.CorrContext);

for i=1:length(gui.aChanPanel)
    if gui.aChanPanel(i)==panel
        gui.aChanPanel(i)=[];
        break;
    end
end
delete(panel)
set(gui.fig,'userdata',gui)
set(gui.aChanPanel,'units','pixels');
for i=1:length(gui.aChanPanel)
    set(gui.aChanPanel(i),'position',[5 5+(i-1)*260 1190 250])
end
p=get(gui.aFig,'position');
set(gui.aFig,'position',[p(1) p(2) 1200 5+260*(length(gui.aChanPanel))])
set(gui.aChanPanel,'units','normalized');
CleanChanLists

screensize=get(0,'ScreenSize');
p=get(gui.aFig,'position');
if (p(2)+p(4))<0
    p(2)=0;
end
set(gui.aFig,'position',p)



function ActivateSpectrum(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');

%Toggle between FFT and Spect
switch obj
    case aChan.aSpect
        set(aChan.aFFT,'value',0)
    case aChan.aFFT
        set(aChan.aSpect,'value',0)
end

set([aChan.aFFTAx,aChan.aFFTPl],'visible','off')
set(aChan.aFFTCap,'visible','off')
set([aChan.aSpectAx, aChan.aSpectIm],'visible','off')
set(aChan.aSpectCap,'visible','off')
    
%Turn off controls for correlation view, time, etc
if get(aChan.aSpect,'value')|get(aChan.aFFT,'value')
    set(aChan.aRTCorrControl,'value',0,'enable','off')
    set(aChan.aRTAnalysis,'value',0,'enable','off')
    sizeAnalAx(aChan.aRTAnalysis,'')
    set([aChan.aTH1Val, aChan.aTH2Val],'enable','off')
    set([aChan.aThresh1, aChan.aThresh2],'value',0,'enable','off')
    analysisThreshSet(aChan.aThresh1,'',gui.aChanPanel(end))
    set(aChan.aWindow,'enable','off')
    analysisThreshSet(aChan.aTH1Val,'',panel)
end

%Activate Spectrogram
if get(aChan.aSpect,'value')
    set(aChan.aSourceAx,'position',[550/1190 50/230 605/1190 165/230])
    set([aChan.aSpectAx, aChan.aSpectIm],'visible','on')
    set(aChan.aSpectCap,'visible','on')
    set(aChan.aSpectIm,'cdata',zeros((3/2)*gui.ai.samplesacquiredfcncount,100,3),...
        'ydata',linspace(0,gui.ai.samplerate/2,(3/2)*gui.ai.samplesacquiredfcncount)/1000,...
        'xdata',(1:100)*0.050)
end

%Activate FFT
if get(aChan.aFFT,'value')
    set(aChan.aSourceAx,'position',[550/1190 50/230 605/1190 165/230])
    set([aChan.aFFTAx,aChan.aFFTPl],'visible','on')
    set(aChan.aFFTCap,'visible','on')
end

%Cleanup if neither
if get(aChan.aSpect,'value')==0&get(aChan.aFFT,'value')==0
    set(aChan.aRTCorrControl,'value',0,'enable','on')
    set(aChan.aRTAnalysis,'value',0,'enable','on')
    sizeAnalAx(aChan.aRTAnalysis,'')
    set([aChan.aThresh1, aChan.aThresh2],'enable','on')
    set(aChan.aWindow,'enable','on')
end

function sizeAnalAx(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(get(obj,'parent'),'userdata');
ThisChanPan=get(obj,'parent');

spaces=1+get(aChan.aRTCorrControl,'value')+get(aChan.aRTAnalysis,'value');
set([aChan.aCWindowT, aChan.aCWindowT],'visible','off')
set([aChan.aCorrRST,aChan.aAnalysisRST],'enable','off')
set([aChan.aCorrPl aChan.aCorrTx],'visible','off')
switch spaces
    case 1
        set(aChan.aSourceAx,'position',gui.aSourcePos(1,:))
        set(aChan.aSourceCap,'position',gui.aSourceCapPos(1,:))
        set(findobj('parent',ThisChanPan,'tag','analysis'),'visible','off')
        set(aChan.aAnalPl,'visible','off')
        set(findobj('parent',ThisChanPan,'tag','correlation'),'visible','off')
        set([aChan.aCorrPl aChan.aCorrTx],'visible','off')
    case 2
        set(aChan.aSourceAx,'position',gui.aSourcePos(2,:))
        set(aChan.aSourceCap,'position',gui.aSourceCapPos(2,:))
        switch get(aChan.aRTCorrControl,'value')
            case 0 %Show Analysis Only
                set(aChan.aAnalAx,'position',gui.aAnalPos(2,:))
                set(aChan.aAnalCap,'position',gui.aAnalCapPos(2,:))
                set(findobj('parent',ThisChanPan,'tag','correlation'),'visible','off')
                set(findobj('parent',ThisChanPan,'tag','analysis'),'visible','on')
                set([aChan.aAnalysisRST],'enable','on')
                set(aChan.aAnalPl,'visible','on')
                set([aChan.aCorrPl aChan.aCorrTx],'visible','off')
                oldAnals=get(aChan.aAnalPl,'userdata');
                AnalysisUpdate(aChan,oldAnals,get(obj,'parent'))
            case 1 %Show Correlation Only
                set(aChan.aCorrAx,'position',gui.aCorrPos(2,:))
                set(aChan.aCCapture,'position',gui.aCorrCapPos(2,:))
                set(findobj('parent',ThisChanPan,'tag','analysis'),'visible','off')
                set(findobj('parent',ThisChanPan,'tag','correlation'),'visible','on')
                set([aChan.aCWindowT, aChan.aCWindowT],'visible','on')                
                set([aChan.aCorrRST],'enable','on')
                set(aChan.aAnalPl,'visible','off')
                set([aChan.aCorrPl aChan.aCorrTx],'visible','on')
        end
    case 3
        set(findobj('parent',ThisChanPan,'tag','correlation'),'visible','on')
        set(findobj('parent',ThisChanPan,'tag','analysis'),'visible','on')
        set(aChan.aCorrAx,'position',gui.aCorrPos(3,:))
        set(aChan.aCCapture,'position',gui.aCorrCapPos(3,:))
        set(aChan.aAnalAx,'position',gui.aAnalPos(3,:))
        set(aChan.aAnalCap,'position',gui.aAnalCapPos(3,:))
        set(aChan.aSourceAx,'position',gui.aSourcePos(3,:))
        set(aChan.aSourceCap,'position',gui.aSourceCapPos(3,:))
        set([aChan.aCorrRST,aChan.aAnalysisRST],'enable','on')
        set(aChan.aAnalPl,'visible','on')
        set([aChan.aCorrPl aChan.aCorrTx],'visible','on')
        oldAnals=get(aChan.aAnalPl,'userdata');
        AnalysisUpdate(aChan,oldAnals,get(obj,'parent'))
end
AnalysisSetting([],[],get(obj,'parent'))

function aTraceSet(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');
times=get(aChan.aSourceTime,'string');
amps=get(aChan.aSourceAmp,'string');
previousTime=max(get(aChan.aSourceAx,'xlim'))/gui.ai.samplerate;
newTime=str2num(times{get(aChan.aSourceTime,'value')});
if previousTime~=newTime
    set(aChan.aSourcePl,'userdata',[]); 
    set(aChan.aSourcePl,'xdata',NaN,'ydata',NaN);
    delete(findobj('tag','gSS07events')); 
    set(aChan.aSourceAx,'userdata',[])
end
    
set(aChan.aSourceTime,'userdata',str2num(times{get(aChan.aSourceTime,'value')}))
set(aChan.aSourceAx,'xlim',[0 str2num(times{get(aChan.aSourceTime,'value')})*gui.ai.samplerate])
set(aChan.aSourceAx,'xticklabel',get(aChan.aSourceAx,'xtick')/gui.ai.samplerate)
set(aChan.aSourceAx,'ylim',str2num(amps{get(aChan.aSourceAmp,'value')}))
aChan.scale=max(str2num(amps{get(aChan.aSourceAmp,'value')}));
set(panel,'userdata',aChan)
set([aChan.aThresh1Pl,aChan.aThresh2Pl],'xdata',get(aChan.aSourceAx,'xlim'))


function AnalScaleScroll(obj,event)
%Scale the axes w/ the data based on mouse scroll
gui=get(findobj('tag','gSS07'),'userdata');
try
    ax=get(gcf,'currentaxes');
    if isempty(ax); return; 
    end
end

aChan=get(get(ax,'parent'),'userdata');
if ax==aChan.aSourceAx
    s=get(aChan.aSourceAmp,'string');
    val=get(aChan.aSourceAmp,'value')-event.VerticalScrollCount;
    maxVal=length(get(aChan.aSourceAmp,'string'));
    if val>maxVal; val=maxVal; end
    if val<1; val=1; end
    set(aChan.aSourceAmp,'value',val);
    set(aChan.aSourceAx,'ylim',str2num(s{val}))
    aTraceSet([],'',get(gca,'parent'))
elseif ax==aChan.aCorrAx
    set(ax,'ylim',get(ax,'ylim')*(2^event.VerticalScrollCount))
    ylim=get(aChan.aCorrAx,'ylim');
    for i=1:length(aChan.aCorrTx)
        info=get(aChan.aCorrTx(i),'userdata');
        set(aChan.aCorrTx(i),'position',[0 1 0]*info.valCount/(10/diff(ylim)))
        ydat=get(info.plot,'ydata');  
        if ~isnan(ydat); set(info.plot,'ydata',info.raw+info.valCount/(10/diff(ylim))); end
        info.offset=info.valCount/(10/diff(ylim));
        set(aChan.aCorrTx(i),'userdata',info);
    end
end

function AnalScaleScrollKey(obj,event)
%Scale the axes w/ the data based on keypress
if ~(strcmpi(event.Character,'+')|strcmpi(event.Character,'-'))
    return
end
gui=get(findobj('tag','gSS07'),'userdata');
try
    ax=get(gcf,'currentaxes');
    if isempty(ax); return; 
    end
end
event.Character
aChan=get(get(ax,'parent'),'userdata');
switch event.Character
    case '+'
        dir=1;
    case '-'
        dir=-1;
end
if ax==aChan.aSourceAx
    s=get(aChan.aSourceAmp,'string');
    val=get(aChan.aSourceAmp,'value')+dir;
    maxVal=length(get(aChan.aSourceAmp,'string'));
    if val>maxVal; val=maxVal; end
    if val<1; val=1; end
    set(aChan.aSourceAmp,'value',val);
    set(aChan.aSourceAx,'ylim',str2num(s{val}))
    aTraceSet([],'',get(gca,'parent'))
elseif ax==aChan.aCorrAx
    set(ax,'ylim',get(ax,'ylim')*(2^-dir))
    ylim=get(aChan.aCorrAx,'ylim');
    for i=1:length(aChan.aCorrTx)
        info=get(aChan.aCorrTx(i),'userdata');
        set(aChan.aCorrTx(i),'position',[0 1 0]*info.valCount/(10/diff(ylim)))
        ydat=get(info.plot,'ydata');  
        if ~isnan(ydat); set(info.plot,'ydata',info.raw+info.valCount/(10/diff(ylim))); end
        info.offset=info.valCount/(10/diff(ylim));
        set(aChan.aCorrTx(i),'userdata',info);
    end
end


%Set Thresholds in Analysis Mode
function analysisThreshSet(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');

%insert threshold error checking here
thresh1=str2double(get(aChan.aTH1Val,'string'));
if isnan(thresh1) 
    thresh1=get(aChan.aTH1Val,'userdata');
    set(aChan.aTH1Val,'string',num2str(thresh1))
end
set(aChan.aTH1Val,'userdata',thresh1)

thresh2=str2double(get(aChan.aTH2Val,'string'));
if isnan(thresh2) 
    thresh2=get(aChan.aTH2Val,'userdata');
    set(aChan.aTH2Val,'string',num2str(thresh2))
end
set(aChan.aTH2Val,'userdata',thresh2)

if get(aChan.aThresh1,'value')
    set(aChan.aThresh1Pl,'visible','on','xdata',get(aChan.aSourceAx,'xlim'),'ydata',...
        [1 1]*str2double(get(aChan.aTH1Val,'string')))
    set(aChan.aTH1Val,'enable','on')
else
    set(aChan.aThresh1Pl,'visible','off')
    set(aChan.aTH1Val,'enable','off')
end

if get(aChan.aThresh2,'value')
    set(aChan.aThresh2Pl,'visible','on','xdata',get(aChan.aSourceAx,'xlim'),'ydata',...
        [1 1]*str2double(get(aChan.aTH2Val,'string')))
    set(aChan.aTH2Val,'enable','on')
else
    set(aChan.aThresh2Pl,'visible','off')
    set(aChan.aTH2Val,'enable','off')
end

%Drag Trheshold Traces for Analysis
function aDragThresh(obj,event,nT,panel)
aChan=get(panel,'userdata');

set(gcf,'windowbuttonmotionfcn',{@threshMotion,nT,aChan,panel})

function threshMotion(obj,event,nT,aChan,panel)
a=get(aChan.aSourceAx,'currentpoint');
ylim=get(aChan.aSourceAx,'ylim');

if (a(1,2)>max(ylim))||(a(1,2)<min(ylim)); return; end
val=round(a(1,2)*10000)/10000;
if nT==1
    set(aChan.aTH1Val,'userdata',val,'string',num2str(val))
else
    set(aChan.aTH2Val,'userdata',val,'string',num2str(val))
end

analysisThreshSet([],'',panel)

function CorrHideLabels(obj,event,panel)
aChan=get(panel,'userdata');
switch get(obj,'checked')
    case 'on'
        set(obj,'checked','off')
        set(aChan.aCorrTx,'visible','on')
    case 'off'
        set(obj,'checked','on')
        set(aChan.aCorrTx,'visible','off')
end

function CorrCenterEvent(obj,event,panel)
aChan=get(panel,'userdata');
if strcmpi(get(obj,'checked'),'on'); 
    set(gcbo,'checked','off'); 
    set(aChan.aCorrAx,'xticklabelmode','auto')
else
    set(gcbo,'checked','on'); 
    set(aChan.aCorrAx,'xticklabel',get(aChan.aCorrAx,'xtick')-get(aChan.aCWindow,'userdata')/2)
end
CorrClear(obj,event,panel)
    

%Activate a single Correlation Trace
function aCorrSetup(obj,event,panel,ChanID,ty,ChanName)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');
if strcmpi(get(obj,'checked'),'off'); set(obj,'checked','on'); else set(obj,'checked','off'); end
for i=1:length(aChan.aCorrPl)
    info=get(aChan.aCorrTx(i),'userdata');
    if isempty(info); continue; end
    if strcmpi(info.name,get(obj,'label'))
        delete(aChan.aCorrPl(i))
        delete(aChan.aCorrTx(i))
        delete(aChan.aCorrTxContext(i))
        aChan.aCorrPl(i)=[];
        aChan.aCorrTx(i)=[];
        aChan.aCorrTxContext(i)=[];
        break
    end
end
if strcmpi(get(obj,'checked'),'off'); set(panel,'userdata',aChan); return; end

%Text object userdata contains info 
info.ChanID=ChanID;
if ty==1; info.ChanID=gui.aChanPanel(ChanID); end
if ty==0; info.ChanID=gui.pl(ChanID); end
info.ChanName=ChanName;
info.name=get(obj,'Label');
info.offset=0;
info.raw=[];
info.type=ty;
info.count=0;
info.spillover=[];
info.time=cputime;
info.valCount=0;

aChan.aCorrPl(end+1)=line([NaN],[NaN],'parent',aChan.aCorrAx,'userdata',[NaN],'linewidth',2);
info.plot=aChan.aCorrPl(end);
info.panel=panel;

aChan.aCorrTx(end+1)=text(0,0,info.name,'color','k','parent',aChan.aCorrAx,'fontweight','bold','userdata',info);
set(aChan.aCorrTx(end),'buttondownfcn',{@aCorrDragTrace,aChan.aCorrTx(end),panel})
aChan.aCorrTxContext(end+1)=uicontextmenu('parent',ancestor(aChan.aCorrTx(end),'figure'));
set(aChan.aCorrTx(end),'UIContextMenu',aChan.aCorrTxContext(end))
uimenu(aChan.aCorrTxContext(end),'Label','Change Trace Color','callback',{@CorrTraceColor,aChan.aCorrPl(end),aChan.aCorrTx(end)})

set(panel,'userdata',aChan)
wind=floor(get(aChan.aCWindow,'userdata')*gui.ai.samplerate);
set(aChan.aCorrAx,'xlim',[0 wind-1]/gui.ai.samplerate)
if strcmpi(get(aChan.aHideText,'checked'),'on'); set(aChan.aCorrTx,'visible','off'); end
set(aChan.aCorrAx,'xticklabelmode','auto')
if strcmpi(get(aChan.aCorrCenter,'checked'),'on');
     set(aChan.aCorrAx,'xticklabel',get(aChan.aCorrAx,'xtick')-get(aChan.aCWindow,'userdata')/2)
end

%Set Correlation Trace Color
function CorrTraceColor(obj,event,pl,tx)
c=uisetcolor('Select a Trace Color');
if length(c)>1
    set(pl,'color',c)
end

function CorrClear(obj,event,panel)
aChan=get(panel,'userdata');

if obj==aChan.aCWindow
    if isnan(str2double(get(obj,'string')))|str2double(get(obj,'string'))<=0;
        set(obj,'string',num2str(get(obj,'userdata'))); 
    else
        set(obj,'userdata',str2double(get(obj,'string'))); 
    end
    set(aChan.aCorrAx,'xlim',[0 get(obj,'userdata')])
end

if obj==aChan.aCResetTime
    if isnan(str2double(get(obj,'string')))|str2double(get(obj,'string'))<0;
        set(obj,'string',num2str(get(obj,'userdata'))); 
    else
        set(obj,'userdata',str2double(get(obj,'string'))); 
    end
end

for i=1:length(aChan.aCorrTx)
    info=get(aChan.aCorrTx(i),'userdata');
    info.raw=[];
    info.count=0;
    info.spillover=[];
    info.time=cputime;
    set(aChan.aCorrTx(i),'userdata',info)
end

set(aChan.aCorrAx,'xticklabelmode','auto')
if strcmpi(get(aChan.aCorrCenter,'checked'),'on');
     set(aChan.aCorrAx,'xticklabel',get(aChan.aCorrAx,'xtick')-get(aChan.aCWindow,'userdata')/2)
end

%Functions to drag individual correlation traces around
function aCorrDragTrace(obj,event,tx,panel)
aChan=get(panel,'userdata');
set(gcf,'windowbuttonmotionfcn',{@DragCorr,aChan,tx})

function DragCorr(obj,event,aChan,tx)
a=get(aChan.aCorrAx,'currentpoint');
ylim=get(aChan.aCorrAx,'ylim');
if (a(1,2)>max(ylim))||(a(1,2)<min(ylim)); return; end
val=round(a(1,2)*10/diff(ylim));
info=get(tx,'userdata');
set(tx,'position',[0 val/(10/diff(ylim)) 0]);
ydat=get(info.plot,'ydata');  
if ~isnan(ydat); set(info.plot,'ydata',info.raw+val/(10/diff(ylim))); end
info.offset=val/(10/diff(ylim));
info.valCount=val;
set(tx,'userdata',info)

function aRTCalc(data,time,gui,sRate,event)
s=get(gui.ChanList,'string');
try
    gui.aChanPanel;
catch
    return;
end

for i=1:length(gui.aChanPanel)
    try aChan=get(gui.aChanPanel(i),'userdata'); catch; return; end
    try ChanName=aChan.name; catch; return; end
    s(1);
    for j=1:length(s); if strcmpi(s{j},ChanName); ind=j; end; end;

    Span=get(aChan.aSourceTime,'userdata');
    if get(aChan.Pause,'value'); 
        set(aChan.aSourcePl,'userdata',[])
        continue; 
    end
      
    if sRate~=aChan.filt.sRate
        %reset filters to new samplerate if sRate has changed
        aRTFiltCalc([],'',gui.aChanPanel(i))
        aChan=get(gui.aChanPanel(i),'userdata');
        set(aChan.aSourceAx,'userdata',[]);
        for j=1:length(gui.aChanPanel)
            aChanTemp=get(gui.aChanPanel(j),'userdata');
            set(aChanTemp.aSourcePl,'userdata',[],'ydata',[])
            aTraceSet([],'',gui.aChanPanel(j))
        end
        
    end
    
    oldTime=get(aChan.aSourcePl,'userdata');
    if isempty(oldTime); oldTime=0; set(aChan.aSourcePl,'ydata',nan,'xdata',nan); end
    
    oldData=get(aChan.aSourcePl,'ydata')';
    if isnan(oldData); oldData=[]; end
    
    %Buffer data for clean filter application at left edge
    predata=get(aChan.aHP,'userdata');
    
    %Invert Data
    if strcmpi(get(aChan.aInvert,'checked'),'on'); data(:,ind)=-data(:,ind); end
    
    %Sliding scope view
    if length(oldData)<(Span*sRate)
        dispdata=[oldData(:); data(:,ind)];
        streaming=0;
    else
        streaming=1;
        dispdata=oldData;
        dispdata(1:(end-length(data(:,ind))))=dispdata((length(data(:,ind))+1):end);
    end
    
    %Apply Filters (High Pass, Low Pass, 60Hz Notch) if Selected
    filtdata=[predata; data(:,ind)];
    if ~(aChan.filt.HP.a==1&aChan.filt.HP.b==1)
        filtdata=filter(aChan.filt.HP.b,aChan.filt.HP.a,filtdata);
    end
    if ~(aChan.filt.LP.a==1&aChan.filt.LP.b==1)
        filtdata=filter(aChan.filt.LP.b,aChan.filt.LP.a,filtdata);
    end
    if ~(aChan.filt.f60.a==1&aChan.filt.f60.b==1)
        filtdata=filter(aChan.filt.f60.b,aChan.filt.f60.a,filtdata);
    end
    
    %Filter Edge Buffer For next loop
    predata(1:length(data(:,ind))*3)=predata((length(data(:,ind))+1):end);
    predata((length(data(:,ind))*3+1):end)=data(:,ind);
    set(aChan.aHP,'userdata',predata)

    %Extract Window Shape both pre and post event
    for j=1:4
        if strcmpi(get(aChan.aWindowCent(j),'checked'),'on')
            WindWidth=j;
        end
    end    
    wind=get(aChan.aWindow,'userdata');
    wind=ceil(wind*sRate/4)*4;
    switch WindWidth
        case 1; prewind=0; postwind=wind;
        case 2; prewind=wind*.25; postwind=wind*.75;
        case 3; prewind=wind*.5; postwind=wind*.5;
        case 4; prewind=wind*.75; postwind=wind*.25;
    end
    thisData=filtdata((end-length(data(:,ind))+1):end);
    dispdata((end-length(data(:,ind))+1):end)=thisData;
    basetime=length(dispdata)-length(data(:,ind))-1;
    
    %shift old overlays back left, delete expired overlays, add new overlays
    oldtraces=get(aChan.aThresh1,'userdata');
    if isempty(oldData); try; delete(oldtraces); end; oldtraces=[]; end
    if streaming==1
        delind=[];
        for j=1:length(oldtraces)
            try xdat=get(oldtraces(j),'xdata');
            catch delind=[delind,j]; continue; end
            xdat=xdat-length(thisData);
            if xdat(end)<0
                delind=[delind,j];
            else
                set(oldtraces(j),'xdata',xdat)
            end
        end
        try delete(oldtraces(delind)); 
        catch
            for j=1:length(oldtraces)
                try delete(oldtraces(j)); end
            end
        end
        oldtraces(delind)=[];
    end
    
    %Threshold current data segment starting where previous loop window ends
    Events=[]; TH=[]; Th=[];
    TH1v=get(aChan.aThresh1,'value');
    TH2v=get(aChan.aThresh2,'value');
    oldEvent=get(aChan.aSourceAx,'userdata');
    old=[];
    
    if TH1v|TH2v
        mData=mean(thisData);
        if TH1v==1&TH2v==0 %only threshold one selected
            TH=get(aChan.aTH1Val,'userdata');
        elseif TH1v==0&TH2v==1 %only threshold two selected
            TH=get(aChan.aTH2Val,'userdata');
        elseif TH1v==1&TH2v==1 %Both thresholds selected
            TH1=get(aChan.aTH1Val,'userdata');
            TH2=get(aChan.aTH2Val,'userdata');
            TH=min(abs([TH1,TH2]));
            Th=max(abs([TH1,TH2]));
        end
       
        %Smart Detect location above or below the mean of the signal
        if TH>mData
            if strcmpi(get(aChan.aRectify,'checked'),'on'); 
                Events=find(abs(thisData)>=TH);
            else
                Events=find(thisData>=TH);
            end
        else
            if strcmpi(get(aChan.aRectify,'checked'),'on'); 
                Events=find(abs(thisData)<=TH);
            else
                Events=find(thisData<=TH);
            end
        end
        
        
       
        %Wipe Events in the window from the oldEvent that overlaps
        %(should only ever be 1 of these)
        if isstruct(oldEvent)
            Events(Events<=oldEvent.samplesremaining)=[];
            oldEvent.data=[oldEvent.data(:);thisData(1:min([length(thisData) oldEvent.samplesremaining]))];
            oldEvent.samplesremaining=wind-length(oldEvent.data);
            if oldEvent.samplesremaining<0
                oldEvent.data=oldEvent.data(1:(end+oldEvent.samplesremaining));
                oldEvent.samplesremaining=0;
            end
            if oldEvent.samplesremaining~=0
                Events=[];
            end
            if oldEvent.samplesremaining==0
                old.data=oldEvent.data;
                old.samplesremaining=0;
                old.threshcross=oldEvent.threshcross;
                old.oldTime=oldEvent.oldTime;
                oldEvent=[];
            end
        end
        %Apply Window to candidate threshold crosses and create an event list
        j=1;
        while j<=length(Events)
            Events(((Events-Events(j))<postwind)&((Events-Events(j))>0))=[];
            j=j+1;
        end
    end
    
    if ~isempty(Events)
        %Event Detected which overflows into previous data range
        %Should only happen at sweep initialization 
        %or when visualizing a very short trace w/ big window, so discard
        if Events(1)<=prewind; 
            if (length(dispdata)-length(thisData))<prewind
                Events(1)=[];
            end
        end
    end
        %Detect an Overflow Event with data we need to save for next loop;
        %Extract it from list of events to process this loop
    if ~isempty(Events)
        j=length(Events);
        if (length(thisData)-Events(j))<=postwind 
            indd=(0:(wind-1))-prewind+Events(j);
            if min(indd)>0
                oldEvent.data=thisData(min(indd):end);
            else
                oldEvent.data=thisData(1:min([max(indd) length(thisData)]));
                oldEvent.data=[dispdata((min(indd):0)+(end-length(data(:,1))));oldEvent.data];
            end
            oldEvent.threshcross=Events(j)+oldTime;
            oldEvent.samplesremaining=wind-length(oldEvent.data); 
            oldEvent.oldTime=oldTime;
            Events(j)=[]; 
        end
    end

    
    %if there is an oldEvent that is ready, process it
    if isstruct(old)
        old.maxamp=max(old.data);
        old.minamp=min(old.data);
        old.maxtime=find(old.data==old.maxamp,1,'first')+old.threshcross-prewind;
        old.mintime=find(old.data==old.minamp,1,'first')+old.threshcross-prewind;
        old.threshcross=old.threshcross;
    end

    %Calculate Max/Mins/times - Reject based on 2 thresholds
    maxtime=zeros(1,length(Events));
    mintime=maxtime;    maxamp=maxtime;    minamp=maxtime;
    threshcross=(Events+oldTime);    
    for j=1:length(Events)
       indd=(0:(wind-1))-prewind+Events(j);
       if min(indd)>0
           eventdata=thisData(indd);
       else
           eventdata=thisData(1:max(indd));
           eventdata=[dispdata((min(indd):0)+(end-length(data(:,1))));eventdata];
       end
       maxamp(j)=max(eventdata);
       maxtime(j)=(find(eventdata==maxamp(j),1,'first')+Events(j)-prewind+oldTime);
       minamp(j)=min(eventdata);
       mintime(j)=(find(eventdata==minamp(j),1,'first')+Events(j)-prewind+oldTime);
    end

    %Reject based on second threshold
    delind=[];
    if ~isempty(Th)
        for j=1:length(Events)
            if maxamp(j)>Th; delind=[delind,j]; end
        end
        Events(delind)=[];
        maxtime(delind)=[];
        mintime(delind)=[];
        maxamp(delind)=[];
        minamp(delind)=[];
        threshcross(delind)=[];
        if isstruct(old)
            if old.maxamp>Th; old=[]; end
        end
    end
    
    fpeaks=zeros(1,length(Events));
    fsum=zeros(1,length(Events));
    
    %Flag for interpolated FFT for higher frequency resolution
    hrsig=0;
    resampRes=20;
    
    %Analyze events (FFT)
    if ~isempty(Events)|isstruct(old)
        if hrsig==0;
            freq=linspace(0,sRate/2,wind/2);
        else
            hrfreq=linspace(0,sRate/2,resampRes*wind/2);
        end
        
        %Calculate a Hamming window for the FFT
        gTx=(0:(wind/2-1))'/(wind-1);
        hamwind=0.55-0.46*cos(2*pi*gTx);
        hamwind=[hamwind;hamwind(end:-1:1)];
        
        for j=1:length(Events)
            indd=(0:(wind-1))-prewind+Events(j);
            if min(indd)>0
                eventdata=thisData(indd);
            else
                eventdata=thisData(1:max(indd));
                eventdata=[dispdata((min(indd):0)+(end-length(data(:,1))));eventdata];
            end
            fsig=abs(fft((eventdata-mean(eventdata)).*hamwind));
            fsig=fsig(1:ceil(end/2));
            if hrsig==0
                ffpeak=find(fsig==max(fsig),1,'first');
                fpeaks(j)=freq(ffpeak);
            else
                hrfsig=resample(fsig,resampRes,1);
                ffpeak=find(hrfsig==max(hrfsig),1,'first');
                fpeak(j)=hrfreq(ffpeak);
            end
            fsum(j)=sum(fsig.^2)/length(fsig);
        end
        if isstruct(old)
            fsig=abs(fft((old.data-mean(old.data)).*hamwind));
            fsig=fsig(1:ceil(end/2));
            if hrsig==0
                ffpeak=find(fsig==max(fsig),1,'first');
                old.fpeaks=freq(ffpeak);
            else
                hrfsig=resample(fsig,resampRes,1);
                ffpeak=find(hrfsig==max(hrfsig),1,'first');
                old.fpeaks=hrfreq(ffpeak);                
            end
            old.fsum=sum(fsig.^2)/length(fsig);
        end   
    end
    
    %maxtime= time of max amplitudes
    %maxamp= peak value in window
    %mintime= time of min amplitudes
    %minamp= min value in window
    %fpeaks= peak frequency component
    %fsum= sum of FFT components (energy density)
    %threshcross= time of threshold cross
    
    %Reject Based on Freq+ED Thresholds
    if get(aChan.aEDenThresh,'value')
        EDTh=get(aChan.aEDenThresh,'userdata');
        if isstruct(old)
            if old.fsum<EDTh(1)|old.fsum>EDTh(2)
                old=[];
            end
        end
        if ~isempty(Events)
            delind=[];
            for j=1:length(Events)
                if fsum(j)<EDTh(1)|fsum(j)>EDTh(2)
                    delind=[delind,j];
                end
            end
            Events(delind)=[];
            maxtime(delind)=[];
            mintime(delind)=[];
            maxamp(delind)=[];
            minamp(delind)=[];
            threshcross(delind)=[];
            fpeaks(delind)=[];
            fsum(delind)=[];
        end
    end
    if get(aChan.aFreqThresh,'value')
        EDTh=get(aChan.aFreqThresh,'userdata');
        if isstruct(old)
            if old.fpeaks<EDTh(1)|old.fpeaks>EDTh(2)
                old=[];
            end
        end
        if ~isempty(Events)
            delind=[];
            for j=1:length(Events)
                if fpeaks(j)<EDTh(1)|fpeaks(j)>EDTh(2)
                    delind=[delind,j];
                end
            end
            Events(delind)=[];
            maxtime(delind)=[];
            mintime(delind)=[];
            maxamp(delind)=[];
            minamp(delind)=[];
            threshcross(delind)=[];
            fpeaks(delind)=[];
            fsum(delind)=[];
        end
    end
    
    %Update Analysis Display
    %Store in a graphical point's userdata, all the analysis parameters
    %Just draw a point based on the two visible parameters at the moment, no concatenation
    
    if get(aChan.aRTAnalysis,'value')==1
        oldAnals=get(aChan.aAnalPl,'userdata');
        if isempty(oldAnals);
            oldAnals.maxtime=[];   oldAnals.maxamp=[];
            oldAnals.mintime=[];   oldAnals.minamp=[];
            oldAnals.fpeaks=[];    oldAnals.fsum=[];
            oldAnals.threshcross=[];
        end
        if isstruct(old)
            oldAnals.maxtime=[oldAnals.maxtime, old.maxtime/sRate];
            oldAnals.maxamp=[oldAnals.maxamp, old.maxamp];
            oldAnals.mintime=[oldAnals.mintime, old.mintime/sRate];
            oldAnals.minamp=[oldAnals.minamp, old.minamp];
            oldAnals.fpeaks=[oldAnals.fpeaks, old.fpeaks];
            oldAnals.fsum=[oldAnals.fsum, old.fsum];
            oldAnals.threshcross=[oldAnals.threshcross, old.threshcross/sRate];
        end
        if ~isempty(Events)
            oldAnals.maxtime=[oldAnals.maxtime, maxtime/sRate];
            oldAnals.maxamp=[oldAnals.maxamp, maxamp];
            oldAnals.mintime=[oldAnals.mintime, mintime/sRate];
            oldAnals.minamp=[oldAnals.minamp, minamp];
            oldAnals.fpeaks=[oldAnals.fpeaks, fpeaks];
            oldAnals.fsum=[oldAnals.fsum, fsum];
            oldAnals.threshcross=[oldAnals.threshcross, threshcross'/sRate];
        end
        if length(oldAnals.threshcross)>2000 %Limit size of data points to between 1000 and 2000 
            oldAnals.maxtime=oldAnals.maxtime(end-1000:end);
            oldAnals.maxamp=oldAnals.maxamp(end-1000:end);
            oldAnals.mintime=oldAnals.mintime(end-1000:end);
            oldAnals.minamp=oldAnals.minamp(end-1000:end);
            oldAnals.fpeaks=oldAnals.fpeaks(end-1000:end);
            oldAnals.fsum=oldAnals.fsum(end-1000:end);
            oldAnals.threshcross=oldAnals.threshcross(end-1000:end);
        end
        set(aChan.aAnalPl,'userdata',oldAnals);
    end
    
    %Call Analysis Visualization Algorithm and pass oldAnals
    if (length(Events)+length(old))>0
        if get(aChan.aRTAnalysis,'value')==1
            yVal=get(aChan.aYval,'value');
            if yVal==8
            elseif yVal==9
            else
                AnalysisUpdate(aChan,oldAnals,gui.aChanPanel(i))
            end
        end
    end
    
    %Call Correlation subfunction w/ completed event list

    %Draw Red Event Overlay
    for j=1:length(Events)
        indd=(0:(wind-1))-prewind+Events(j);
        if min(indd)>0
            eventdata=thisData(indd);
            oldtraces=[oldtraces,line(((0:(wind-1))+Events(j)+basetime-prewind),eventdata,...
                'tag','gSS07events','color','r','parent',aChan.aSourceAx,'UIContextMenu',aChan.DataContext)];    
        else
            eventdata=thisData(1:max(indd));
            eventdata=[dispdata((min(indd):0)+(end-length(data(:,1))));eventdata];
            oldtraces=[oldtraces,line(((0:(wind-1))+Events(j)+basetime-prewind),eventdata,...
                'tag','gSS07events','color','r','parent',aChan.aSourceAx,'UIContextMenu',aChan.DataContext)];    
        end
    end
    if isstruct(old)
        oldtraces=[oldtraces,line((0:(wind-1))-(oldTime-old.threshcross)+length(dispdata)-prewind-length(data(:,ind))-1,old.data,...
            'tag','gSS07events','color','r','parent',aChan.aSourceAx,'UIContextMenu',aChan.DataContext)];    
    end
    set(aChan.aThresh1,'userdata',oldtraces);
    
    %Keep the threshold traces on top of all graphics so it can be dragged
    if ~isempty(Events)
        set([aChan.aThresh1Pl,aChan.aThresh2Pl],'parent',gui.backax)
        set([aChan.aThresh1Pl,aChan.aThresh2Pl],'parent',aChan.aSourceAx)
    end

    %Store Event that could not complete window in this sweep
    set(aChan.aSourceAx,'userdata',oldEvent);    

    %Display Filtered Data
    if streaming==0
        time=(0:(length(dispdata)-1));
        set(aChan.aSourcePl,'xdata',time,'ydata',dispdata,'userdata',oldTime+length(data(:,ind)))
    else
        set(aChan.aSourcePl,'ydata',dispdata,'userdata',oldTime+length(data(:,ind)))
    end
    
    %Output sound if requested
    if get(aChan.Listen,'value')==1
        sig=dispdata((end-length(data(:,ind))+1):end)/aChan.scale;
        OutputUserAudio(sig,gui,sRate)
    end
    
    %Update FFT Display
    if get(aChan.aFFT,'value')
        aft=abs(fft(dispdata));
        aft=aft(1:floor(end/2));
        freq=linspace(0,sRate/2,length(aft));
        set(aChan.aFFTPl,'xdata',freq,'ydata',aft)
        ylim=get(aChan.aFFTAx,'ylim');
        set(aChan.aFFTAx,'ylim',[0 max([max(ylim)*.98 max(aft)])])
    end
   
    %Update Spectrogram Display
    if get(aChan.aSpect,'value')
        %aChan.aSpectIm is the image object, update x,y,cdata w/ current 50ms slot
        cData=get(aChan.aSpectIm,'cdata');
        aft=abs(fft(thisData));
        aft=aft(1:floor(end/2));
        aft=abs(resample(aft,3,1));
        freq=linspace(0,sRate/2,length(aft))/1000;
        
        newCol=aft/max(aft);

        if length(cData(:,1))~=length(newCol)
            cData=zeros(length(newCol),100);
        else
            cData=cData(:,:,1);
            cData(:,(2:end))=cData(:,1:(end-1));
        end
            cData(:,1)=newCol;

        cData=repmat(cData,[1 1 3]);
        set(aChan.aSpectIm,'cdata',cData,'ydata',freq,'xdata',[1:101]*0.050)
        set(aChan.aSpectAx,'ylim',[0 max(freq)],'xlim',[1 101]*0.050)
        
    end
    
    %Gather Correlation info maxtime mintime
    if strcmpi(get(aChan.aCorrTrig(2),'checked'),'on')
        corrInfo(i).Events=Events+basetime-length(dispdata);
        corrInfo(i).oldF=isstruct(old);
        if isstruct(old); corrInfo(i).oldEvent=old.threshcross-old.oldTime+basetime-length(dispdata)+(old.oldTime-oldTime);
        else corrInfo(i).oldEvent=[];
        end
        corrInfo(i).thisData=thisData;
    elseif strcmpi(get(aChan.aCorrTrig(3),'checked'),'on')
        corrInfo(i).Events=maxtime-oldTime+basetime-length(dispdata);
        corrInfo(i).oldF=isstruct(old);
        if isstruct(old); corrInfo(i).oldEvent=(oldTime-old.maxtime)+length(dispdata)-length(data(:,ind))-1;
        else corrInfo(i).oldEvent=[];
        end
        corrInfo(i).thisData=thisData;
    elseif strcmpi(get(aChan.aCorrTrig(4),'checked'),'on')
        corrInfo(i).Events=mintime-oldTime+basetime-length(dispdata);
        corrInfo(i).oldF=isstruct(old);
        if isstruct(old); corrInfo(i).oldEvent=(oldTime-old.mintime)+length(dispdata)-length(data(:,ind))-1;
        else corrInfo(i).oldEvent=[];
        end
        corrInfo(i).thisData=thisData;
    end
end

%Once all trace info is up, carry out correlation
for i=1:length(gui.aChanPanel)
    aChan=get(gui.aChanPanel(i),'userdata');
    wind=floor(get(aChan.aCWindow,'userdata')*sRate);
    center=strcmpi(get(aChan.aCorrCenter,'checked'),'on');
    %Assume time in the corrInfo struct is based on length of plot data vectors
    %Do not process if paused
    if get(aChan.aRTCorrControl,'value')==0; continue; end
    if get(aChan.Pause,'value'); continue; end
    
    for j=1:length(aChan.aCorrTx)        %for each correlation trace that has been added
        info=get(aChan.aCorrTx(j),'userdata');
        eTime=get(aChan.aCResetTime,'userdata');
        %Reset if time loop requirements are met
        if eTime>0
            dTime=cputime-info.time;
            if dTime>eTime
                info.raw=[];
                info.count=0;
                info.spillover=[];
                info.time=cputime;
            end
            set(aChan.aCorrRST,'string',['Corr. Reset (',num2str(ceil(eTime-dTime)),')'])
        else
            set(aChan.aCorrRST,'string','Corr. Reset')
        end
        switch info.type
            case 0 %From raw data trace
                corrTrace=get(info.ChanID,'userdata')';
                %if main fig pause is pressed continue on
                if get(gui.MainFigPause,'value'); continue; end
            case 1 %From analyzed data trace
                oaChan=get(info.ChanID,'userdata');
                corrTrace=get(oaChan.aSourcePl,'ydata');
                %if specific channel id pause is pressed continue on;
                if get(oaChan.Pause,'value'); continue; end
        end
        %Process previous events until they are full length 
        overflow=info.spillover;
        delind=[];
        if ~isempty(info.spillover)
            for k=1:length(overflow)
                ind=(0:(overflow(k).samplesremaining-1))+length(corrTrace)-length(data(:,1))+1;
                if max(ind)>length(corrTrace); ind=min(ind):length(corrTrace); end
                overflow(k).data=[overflow(k).data, corrTrace(ind)];
                overflow(k).samplesremaining=wind-length(overflow(k).data);
                %if complete, draw
                if overflow(k).samplesremaining==0
                    sig=overflow(k).data; sig=sig-sig(1+end/2*center);
                    delind=[delind,k];
                    info.raw=info.raw*(info.count)+sig;
                    info.count=info.count+1;
                    info.raw=info.raw/info.count;
                end
            end
        end
        overflow(delind)=[];
        % assume events are indexed from end back (they will be negative numbers)
        % The last value of corrTrace is time 0
        if isempty(info.raw); info.raw=zeros(1,wind); info.count=0; end
        overlap=[]; ct=0;
        if ~isempty(corrInfo(i).oldEvent)
            ind=(length(corrTrace)+corrInfo(i).oldEvent)+(0:(wind-1))-floor(wind/2)*center;
            if min(ind)<=0; 
                 ct=1; 
            end
            if max(ind)>length(corrTrace); 
                overlap=[overlap,corrInfo(i).oldEvent];
                ct=1; 
            end
            if ct==0
                sig=corrTrace(ind); sig=sig-sig(1+end/2*center);
                info.raw=info.raw*(info.count)+sig;
                info.count=info.count+1;
                info.raw=info.raw/info.count;
            end
        end
        ct=0;
        for k=1:length(corrInfo(i).Events)
            ind=(length(corrTrace)+corrInfo(i).Events(k))+(0:(wind-1))-floor(wind/2)*center;
            if min(ind)<=0; continue; end
            if max(ind)>length(corrTrace); overlap=[overlap,corrInfo(i).Events(k)]; continue; end
            sig=corrTrace(ind); sig=sig-sig(1+end/2*center);
            info.raw=info.raw*(info.count)+sig;
            info.count=info.count+1;
            info.raw=info.raw/info.count;
        end
        %Store Incomplete traces
        if ~isempty(overlap)
            offset=length(overflow);
            p=1;
            for k=1:length(overlap)
                ind=((length(corrTrace)+overlap(k)-floor(wind/2)*center):length(corrTrace));
                if min(ind)<=0; continue; end
                overflow(p+offset).data=corrTrace(ind);
                overflow(p+offset).samplesremaining=wind-length(overflow(k+offset).data);
                p=p+1;
            end
        end
        info.spillover=overflow;
        %Update Trace w/ appropriate offset
        set(info.plot,'ydata',info.raw+info.offset,'xdata',(0:(wind-1))/sRate)
        set(aChan.aCorrTx(j),'userdata',info)
    end
end

function aRTFiltCalc(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
if ishandle(obj)
    if strcmpi(get(obj,'Label'),'60 Hz Notch')
        if strcmpi(get(obj,'checked'),'on')
            set(obj,'checked','off')
        else
            set(obj,'checked','on')
        end
    else
        try
            set(get(obj,'userdata'),'checked','off')
            set(obj,'checked','on')
        end
    end
end
aChan=get(panel,'userdata');

sRate=gui.ai.samplerate;
aChan.filt.sRate=sRate;
predata=zeros(gui.ai.samplesacquiredfcncount*4,1);
set(aChan.aHP,'userdata',predata)

%HP Filter Frequency
fHP=0;
for i=1:length(aChan.aHPMenu)
    if strcmpi(get(aChan.aHPMenu(i),'checked'),'on')
        str=get(aChan.aHPMenu(i),'Label');
        if strcmpi(str,'None')
            fHP=0;
        else
            fHP=str2double(str);
        end
    end
end
%LP Filter Frequency
fLP=0;
for i=1:length(aChan.aLPMenu)
    if strcmpi(get(aChan.aLPMenu(i),'checked'),'on')
        str=get(aChan.aLPMenu(i),'Label');
        if strcmpi(str,'None')
            fLP=0;
        else
            fLP=str2double(str);
        end
    end
end

%Calculate Individual Filters
aChan.filt.BP.a=1; aChan.filt.BP.b=1;
if fHP/(sRate/2)>=1; 
    aRTFiltCalc(aChan.aHPMenu(1),'',panel);
    return; 
elseif fHP==0
    aChan.filt.HP.a=1;
    aChan.filt.HP.b=1;
else
    [aChan.filt.HP.b aChan.filt.HP.a]=butter(3,fHP/(sRate/2),'high');
end
if fLP/(sRate/2)>=1; 
    aRTFiltCalc(aChan.aLPMenu(1),'',panel);
    return; 
elseif fLP==0
    aChan.filt.LP.a=1;
    aChan.filt.LP.b=1;
else
    [aChan.filt.LP.b aChan.filt.LP.a]=butter(3,fLP/(sRate/2),'low');
end    

%Calculate 60Hz Notch
if strcmpi(get(aChan.a60Hz,'Checked'),'on')
    wo = 60/(sRate/2);  bw = wo/15;
    [aChan.filt.f60.b,aChan.filt.f60.a] = iirnotch(wo,bw,-6);
else
    aChan.filt.f60.a=1; aChan.filt.f60.b=1;
end

set(panel,'userdata',aChan)

function AnalysisReset(obj,event,panel)
aChan=get(panel,'userdata');
set(aChan.aAnalPl,'userdata',[],'xdata',[],'ydata',[])

function AnalysisSetting(obj,event,panel)
aChan=get(panel,'userdata');
if get(aChan.aRTAnalysis,'value')==0; return; end

%Setup Y-Axis Properties
set([aChan.aEDenThresh aChan.aFreqThresh],'visible','off')
if get(aChan.aYval,'value')==7; set(aChan.aEDenThresh,'visible','on'); end
if get(aChan.aYval,'value')==6; set(aChan.aFreqThresh,'visible','on'); end
if get(aChan.aYlog,'value')==1
    set(aChan.aAnalAx,'yscale','log')
else
    set(aChan.aAnalAx,'yscale','linear')
end
if get(aChan.aYauto,'value')==1; 
    set(aChan.aAnalAx,'ylimmode','auto')
    set([aChan.Ymin, aChan.Ymax],'enable','off')
    ylim=get(aChan.aAnalAx,'ylim');
    set(aChan.Ymin,'string',num2str(ylim(1)),'userdata',ylim(1))
    set(aChan.Ymax,'string',num2str(ylim(2)),'userdata',ylim(2))
else
    set(aChan.aAnalAx,'ylimmode','manual')
    set([aChan.Ymin, aChan.Ymax],'enable','on')
    ylima=str2double(get(aChan.Ymin,'string'));
    if isnan(ylima); ylima=get(aChan.Ymin,'userdata'); end
    ylimb=str2double(get(aChan.Ymax,'string'));
    if isnan(ylimb); ylimb=get(aChan.Ymax,'userdata'); end
    ylim=sort([ylima ylimb]);
    set(aChan.Ymin,'string',num2str(ylim(1)),'userdata',ylim(1))
    set(aChan.Ymax,'string',num2str(ylim(2)),'userdata',ylim(2))
    set(aChan.aAnalAx,'ylim',ylim)
end

%FFT Visualization Mode
if get(aChan.aYval,'value')==8
    set(aChan.aAnalPl,'xdata',[],'ydata',[])
end
%Spectrogram
if get(aChan.aYval,'value')==9
    set(aChan.aAnalPl,'xdata',[],'ydata',[])
end

%X-axis Properties
if get(aChan.aXval,'value')==1; %If on auto-scroll time view
    set([aChan.aXwidthT aChan.aXwidth],'visible','on')
    set([aChan.Xmin aChan.Xmax],'visible','off')
    set(aChan.aXauto,'value',1,'enable','off')
    set(aChan.aXlog,'enable','off','value',0)
    s=get(aChan.aXwidth,'string');
    set(aChan.aXwidth,'userdata',str2double(s{get(aChan.aXwidth,'value')}))
    set(aChan.aAnalAx,'xscale','linear')
else
    set([aChan.aXwidthT aChan.aXwidth],'visible','off')
    set([aChan.Xmin aChan.Xmax],'visible','on')
    set(aChan.aXauto,'enable','on')
    set(aChan.aXlog,'enable','on')

    if get(aChan.aXlog,'value')==1
        set(aChan.aAnalAx,'xscale','log')
    else
        set(aChan.aAnalAx,'xscale','linear')
    end
end
if get(aChan.aXauto,'value')==1; 
    set(aChan.aAnalAx,'xlimmode','auto')
    set([aChan.Xmin, aChan.Xmax],'enable','off')
    xlim=get(aChan.aAnalAx,'xlim');
    set(aChan.Xmin,'string',num2str(xlim(1)),'userdata',xlim(1))
    set(aChan.Xmax,'string',num2str(xlim(2)),'userdata',xlim(2))
else
    set(aChan.aAnalAx,'xlimmode','manual')
    set([aChan.Xmin, aChan.Xmax],'enable','on')
    xlima=str2double(get(aChan.Xmin,'string'));
    if isnan(xlima); xlima=get(aChan.Xmin,'userdata'); end
    xlimb=str2double(get(aChan.Xmax,'string'));
    if isnan(xlimb); xlimb=get(aChan.Xmax,'userdata'); end
    xlim=sort([xlima xlimb]);
    set(aChan.Xmin,'string',num2str(xlim(1)),'userdata',xlim(1))
    set(aChan.Xmax,'string',num2str(xlim(2)),'userdata',xlim(2))
    set(aChan.aAnalAx,'xlim',xlim)
end

EDT=findobj('tag',['gEDenThresh',get(panel,'title')]);
PFT=findobj('tag',['gFreqThresh',get(panel,'title')]);
set([EDT,PFT],'visible','off')

if get(aChan.aYval,'value')==7
    set(EDT,'visible','on')
end
if get(aChan.aYval,'value')==6
    set(PFT,'visible','on')
end

analysis=get(aChan.aAnalPl,'userdata');

function AnalysisUpdate(aChan,analysis,panel)
%maxtime= time of max amplitudes
%maxamp= peak value in window
%mintime= time of min amplitudes
%minamp= min value in window
%fpeaks= peak frequency component
%fsum= sum of FFT components (energy density)
%threshcross= time of threshold cross

if isempty(analysis); 
    set(aChan.aAnalPl,'xdata',[],'ydata',[])
    return;
else
    if length(analysis.threshcross)==0
        set(aChan.aAnalPl,'xdata',[],'ydata',[])
        return
    end
end

if length(analysis.threshcross)>500
    analysis.maxtime=analysis.maxtime(end-500:end);
    analysis.maxamp=analysis.maxamp(end-500:end);
    analysis.mintime=analysis.mintime(end-500:end);
    analysis.minamp=analysis.minamp(end-500:end);
    analysis.fpeaks=analysis.fpeaks(end-500:end);
    analysis.fsum=analysis.fsum(end-500:end);
    analysis.threshcross=analysis.threshcross(end-500:end);
end

% ('Event Time','Rate','Interval','Amplitude(Min)','Amplitude(Max)','FFT Peak','FFT Sum')
switch get(aChan.aXval,'value')
    case 1; xdat=analysis.threshcross;
    case 2; 
            if length(analysis.threshcross)<2;
                xdat=nan*ones(length(analysis.threshcross),1);
            else xdat=1./diff(analysis.threshcross(:));
            end
    case 3;
            if length(analysis.threshcross)<2;
                xdat=nan*ones(length(analysis.threshcross),1);
            else xdat=diff(analysis.threshcross(:));
            end
    case 4; xdat=analysis.minamp;
    case 5; xdat=analysis.maxamp;
    case 6; xdat=analysis.fpeaks;
    case 7; xdat=analysis.fsum;
end
switch get(aChan.aYval,'value')
    case 1; ydat=analysis.threshcross;
    case 2; 
            if length(analysis.threshcross)<2; 
                ydat=nan*ones(length(analysis.threshcross),1);
            else ydat=1./diff(analysis.threshcross(:));
            end
    case 3; 
            if length(analysis.threshcross)<2; 
                ydat=nan*ones(length(analysis.threshcross),1); 
            else ydat=diff(analysis.threshcross(:));
            end
    case 4; ydat=analysis.minamp;
    case 5; ydat=analysis.maxamp;
    case 6; ydat=analysis.fpeaks;
    case 7; ydat=analysis.fsum;
end
if length(xdat)>length(ydat);  xdat=xdat(2:end);   end
if length(ydat)>length(xdat);  ydat=ydat(2:end);   end

set(aChan.aAnalPl,'xdata',xdat,'ydata',ydat)

if get(aChan.aXval,'value')==1
        set(aChan.aAnalAx,'xlim',[-1*get(aChan.aXwidth,'userdata') 0]+analysis.threshcross(end)); 
end


if get(aChan.aEDenThresh,'value')
    set(findobj('tag',['gEDenThresh',get(panel,'title')]),'xdata',get(aChan.aAnalAx,'xlim'))
end
if get(aChan.aFreqThresh,'value')
    set(findobj('tag',['gFreqThresh',get(panel,'title')]),'xdata',get(aChan.aAnalAx,'xlim'))
end

    
% Activate or deactivate Energy Density Thresholds
function aEDenThresh(obj,event,ax,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');

if get(aChan.aEDenThresh,'value')
    %Draw Results Threshold lines
    ylim=get(ax,'ylim');    
    xlim=get(ax,'xlim');    
    line(xlim,[1 1]*(.95*diff(ylim)+min(ylim)),'color','k','marker','.',...
        'parent',ax,'tag',['gEDenThresh',get(panel,'title')],'userdata',1,...
        'buttondownfcn',{@gEDenThreshDrag,1,aChan,panel})
    line(xlim,[1 1]*(.05*diff(ylim)+min(ylim)),'color','k','marker','.',...
        'parent',ax,'tag',['gEDenThresh',get(panel,'title')],'userdata',2,...
        'buttondownfcn',{@gEDenThreshDrag,2,aChan,panel})
    set(aChan.aEDenThresh,'userdata',sort([0.05 0.95]*(diff(ylim)+min(ylim))))
else
    %delete Results Thresholds
    delete(findobj('tag',['gEDenThresh',get(panel,'title')]))
end

%Drag Energy Density Threshold levels
function gEDenThreshDrag(obj,event,nT,aChan,panel)
set(gcf,'windowbuttonmotionfcn',{@gEDenThreshMotion,nT,aChan,panel})

function gEDenThreshMotion(obj,event,nT,aChan,panel)
a=get(aChan.aAnalAx,'currentpoint');
ylim=get(aChan.aAnalAx,'ylim');

if a(1,2)>max(ylim)|a(1,2)<min(ylim); return; end
val=round(a(1,2)*10000)/10000;
if nT==1
    set(findobj('tag',['gEDenThresh',get(panel,'title')],'userdata',1),'ydata',[1 1]*val)
else
    set(findobj('tag',['gEDenThresh',get(panel,'title')],'userdata',2),'ydata',[1 1]*val)    
end
thresh1=mean(get(findobj('tag',['gEDenThresh',get(panel,'title')],'userdata',1),'ydata'));
thresh2=mean(get(findobj('tag',['gEDenThresh',get(panel,'title')],'userdata',2),'ydata'));
set(aChan.aEDenThresh,'userdata',sort([thresh1 thresh2]))

% Activate or deactivate Frequency Thresholds
function aFreqThresh(obj,event,ax,panel)
gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');

if get(aChan.aFreqThresh,'value')
    %Draw Results Threshold lines
    ylim=get(ax,'ylim');    
    xlim=get(ax,'xlim');    
    line(xlim,[1 1]*(.95*diff(ylim)+min(ylim)),'color','k','marker','.',...
        'parent',ax,'tag',['gFreqThresh',get(panel,'title')],'userdata',1,...
        'buttondownfcn',{@gFreqThreshDrag,1,aChan,panel})
    line(xlim,[1 1]*(.05*diff(ylim)+min(ylim)),'color','k','marker','.',...
        'parent',ax,'tag',['gFreqThresh',get(panel,'title')],'userdata',2,...
        'buttondownfcn',{@gFreqThreshDrag,2,aChan,panel})
    set(aChan.aFreqThresh,'userdata',sort([0.05 0.95]*(diff(ylim)+min(ylim))))
else
    %delete Results Thresholds
    delete(findobj('tag',['gFreqThresh',get(panel,'title')]))
end

%Drag Frequency Threshold levels
function gFreqThreshDrag(obj,event,nT,aChan,panel)
set(gcf,'windowbuttonmotionfcn',{@gFreqThreshMotion,nT,aChan,panel})

function gFreqThreshMotion(obj,event,nT,aChan,panel)
a=get(aChan.aAnalAx,'currentpoint');
ylim=get(aChan.aAnalAx,'ylim');

if a(1,2)>max(ylim)|a(1,2)<min(ylim); return; end
val=round(a(1,2)*10000)/10000;
if nT==1
    set(findobj('tag',['gFreqThresh',get(panel,'title')],'userdata',1),'ydata',[1 1]*val)
else
    set(findobj('tag',['gFreqThresh',get(panel,'title')],'userdata',2),'ydata',[1 1]*val)    
end
thresh1=mean(get(findobj('tag',['gFreqThresh',get(panel,'title')],'userdata',1),'ydata'));
thresh2=mean(get(findobj('tag',['gFreqThresh',get(panel,'title')],'userdata',2),'ydata'));
set(aChan.aFreqThresh,'userdata',sort([thresh1 thresh2]))


%load a graph displaying the filter transfer function
function dispFiltSpect(obj,event,panel)
aRTFiltCalc([],[],panel) %Update Filters if sRate changed

gui=get(findobj('tag','gSS07'),'userdata');
aChan=get(panel,'userdata');
sRate=aChan.filt.sRate;


N=1000;
% [afH,afW]=freqz(AFfilt.b,AFfilt.a,N);
[lpH,lpW]=freqz(aChan.filt.LP.b,aChan.filt.LP.a,N);
[hpH,hpW]=freqz(aChan.filt.HP.b,aChan.filt.HP.a,N);
[nfH,nfW]=freqz(aChan.filt.f60.b,aChan.filt.f60.a,N);

FullSpect=lpH.*hpH.*nfH;
W=lpW*sRate/(2*pi);

figure('numbertitle','off','name','Filter Transfer Function');
subplot(2,1,1)
plot(W,20*log10(abs(FullSpect)),'linewidth',2)
set(gca,'xscale','log','ylim',[-30 1],'ygrid','on','xgrid','off')
ylabel('Filter Amplitude (dB)')
title('Filter Transfer Function')

subplot(2,1,2)
plot(W,angle(FullSpect))
set(gca,'xscale','log','ygrid','on','xgrid','off')
ylabel('Filter Phase (Rad)')
xlabel('Frequency (Hz)')





function gPropEdit(varargin)
%Property Editor for an arbitrary window
% Meant to be Called from a figure menu button such that GCF is the figure
% containing the objects to edit.  Design intends one axes in the figure
% with an arbitrary number of lines

% Error correction on entry in edit boxes with numbers
% Sort axes ranges and orient them based on min/max
% Apply immediately?

thisFig=gcf;
legend off
delete(findobj('name',['Property Editor for Figure ',num2str(gcf)]))
set(thisFig,'deletefcn',['delete(findobj(''name'',''','Property Editor for Figure ',num2str(gcf),'''))'])

edit.targetFig=thisFig;

if nargin==0;
    edit.ax=findobj('parent',thisFig,'type','axes');
elseif nargin==1
    edit.ax=findobj('parent',thisFig,'type','axes','tag',varargin{1});
end

edit.pl=findobj('parent',edit.ax,'type','line');



if length(edit.ax)~=1; return; end
if isempty(findobj('parent',edit.ax,'type','patch'))
    if length(edit.pl)==0; return; end 
end

if nargin==1
    edit.pl=[];
end

edit.fig=figure('numbertitle','off','name',['Property Editor for Figure ',...
    num2str(thisFig)],'menubar','none','position',[100 100 400 300]);

%Edit Axes
%Axis Title, Background and Foreground Colors
%X,Y Grid, Box
%X/Y-axis: Limits (auto/manual), Label, Scale, reverse

edit.axpanel=uipanel('title','Axes Properties','units','normalized',...
    'position',[.01 .55 .98 .42],'backgroundcolor',get(gcf,'color'));

uicontrol('parent',edit.axpanel,'style','text','backgroundcolor',...
    get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
    [.77 .8 .15 .1],'string','Axes Limits','horizontalalignment','right')

uicontrol('parent',edit.axpanel,'style','text','backgroundcolor',...
    get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
    [.02 .8 .15 .1],'string','Axis Title','horizontalalignment','right')
edit.axTitle=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.2 .77 .4 .17],'string','',...
    'horizontalalignment','left','callback','spikeHound(''gUpdateGraphics'')');
set(edit.axTitle,'string',get(get(edit.ax,'title'),'string'))
edit.axBackColor=uicontrol('parent',edit.axpanel,'style','pushbutton',...
    'units','normalized','position',[.62 .77 .08 .17],'string','Back',...
    'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
set(edit.axBackColor,'backgroundcolor',get(edit.ax,'color'))
if sum(get(edit.ax,'color'))/3>.5
    set(edit.axBackColor,'foregroundcolor','k')
else
    set(edit.axBackColor,'foregroundcolor','w')
end


uicontrol('parent',edit.axpanel,'style','text','backgroundcolor',...
    get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
    [.02 .6 .15 .1],'string','X-Label','horizontalalignment','right')
edit.axXLabel=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.2 .57 .3 .17],'string','',...
    'horizontalalignment','left','callback','spikeHound(''gUpdateGraphics'')');
set(edit.axXLabel,'string',get(get(edit.ax,'xlabel'),'string'))
edit.axXColor=uicontrol('parent',edit.axpanel,'style','pushbutton',...
    'units','normalized','position',[.52 .57 .08 .17],'string','X',...
    'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
set(edit.axXColor,'backgroundcolor',get(edit.ax,'xcolor'))
if sum(get(edit.ax,'xcolor'))/3>.5
    set(edit.axXColor,'foregroundcolor','k')
else
    set(edit.axXColor,'foregroundcolor','w')
end

edit.axXLog=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.61 .57 .12 .17],'string','Log','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'xscale'),'log');
    set(edit.axXLog,'value',1)
else
    set(edit.axXLog,'value',0)
end
edit.axXlow=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.72 .57 .12 .17],'string','','enable','off','callback','spikeHound(''gUpdateGraphics'')');
edit.axXhigh=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.87 .57 .12 .17],'string','','enable','off','callback','spikeHound(''gUpdateGraphics'')');
set(edit.axXlow,'string',min(get(edit.ax,'xlim')),'userdata',min(get(edit.ax,'xlim')))
set(edit.axXhigh,'string',max(get(edit.ax,'xlim')),'userdata',max(get(edit.ax,'xlim')))


uicontrol('parent',edit.axpanel,'style','text','backgroundcolor',...
    get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
    [.02 .4 .15 .1],'string','Y-Label','horizontalalignment','right')
edit.axYLabel=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.2 .37 .3 .17],'string','',...
    'horizontalalignment','left','callback','spikeHound(''gUpdateGraphics'')');
set(edit.axYLabel,'string',get(get(edit.ax,'ylabel'),'string'))
edit.axYColor=uicontrol('parent',edit.axpanel,'style','pushbutton',...
    'units','normalized','position',[.52 .37 .08 .17],'string','Y',...
    'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
set(edit.axYColor,'backgroundcolor',get(edit.ax,'ycolor'))
if sum(get(edit.ax,'ycolor'))/3>.5
    set(edit.axYColor,'foregroundcolor','k')
else
    set(edit.axYColor,'foregroundcolor','w')
end
edit.axYLog=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.61 .37 .12 .17],'string','Log','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'xscale'),'log');
    set(edit.axYLog,'value',1)
else
    set(edit.axYLog,'value',0)
end
edit.axYlow=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.72 .37 .12 .17],'string','','enable','off','callback','spikeHound(''gUpdateGraphics'')');
edit.axYhigh=uicontrol('parent',edit.axpanel,'style','edit','backgroundcolor','w',...
    'units','normalized','position',[.87 .37 .12 .17],'string','','enable','off','callback','spikeHound(''gUpdateGraphics'')');
set(edit.axYlow,'string',min(get(edit.ax,'ylim')),'userdata',min(get(edit.ax,'ylim')))
set(edit.axYhigh,'string',max(get(edit.ax,'ylim')),'userdata',max(get(edit.ax,'ylim')))


edit.axXgrid=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.3 .17 .15 .1],'string','X Grid','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'xgrid'),'on');
    set(edit.axXgrid,'value',1)
else
    set(edit.axXgrid,'value',0)
end

edit.axYgrid=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.3 .05 .15 .1],'string','Y Grid','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'ygrid'),'on');
    set(edit.axYgrid,'value',1)
else
    set(edit.axYgrid,'value',0)
end

edit.axXLimAuto=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.52 .17 .27 .1],'string','Auto X-Limits','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'xlimmode'),'auto');
    set(edit.axXLimAuto,'value',1)
else
    set(edit.axXLimAuto,'value',0)
    set([edit.axXlow edit.axXhigh],'enable','on')
end

edit.axYLimAuto=uicontrol('parent',edit.axpanel,'style','checkbox','backgroundcolor',get(edit.axpanel,'backgroundcolor'),...
    'units','normalized','position',[.52 .05 .27 .1],'string','Auto Y-Limits','callback','spikeHound(''gUpdateGraphics'')');
if strcmpi(get(edit.ax,'ylimmode'),'auto');
    set(edit.axYLimAuto,'value',1)
else
    set(edit.axYLimAuto,'value',0)
    set([edit.axYlow edit.axYhigh],'enable','on')
end



%Edit Lines
%Line Style, Line Color, Line Size
%Marker Style, Marker Color (edge/face), Marker Size
%Visible on/off
%bring forward - move back

for i=1:length(edit.pl)
    edit.plpanel(i)=uipanel('title',['Plot ',num2str(i),' Properties'],...
        'units','normalized','position',[.01 .1 .98 .42],...
        'backgroundcolor',get(gcf,'color'),'visible','off');

    uicontrol('parent',edit.plpanel(i),'style','text','backgroundcolor',...
        get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
        [.1 .87 .15 .1],'string','Line')
    edit.plstyle(i)=uicontrol('parent',edit.plpanel(i),'style','popupmenu','backgroundcolor',...
        'w','units','normalized','callback','spikeHound(''gUpdateGraphics'')','position',...
        [.07 .75 .2 .1],'string',strvcat('-','--',':','-.','none'),'horizontalalignment','right');
    switch get(edit.pl(i),'linestyle')
        case '-'; set(edit.plstyle(i),'value',1);
        case '--'; set(edit.plstyle(i),'value',2);
        case ':'; set(edit.plstyle(i),'value',3);
        case '-.'; set(edit.plstyle(i),'value',4);
        case 'none'; set(edit.plstyle(i),'value',5);
    end
    
    edit.plcolor(i)=uicontrol('parent',edit.plpanel(i),'style','pushbutton',...
        'units','normalized','position',[.09 .4 .15 .2],'string','Color',...
        'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
    set(edit.plcolor(i),'backgroundcolor',get(edit.pl(i),'color'))
    if sum(get(edit.pl(i),'color'))/3>.5
        set(edit.plcolor(i),'foregroundcolor','k')
    else
        set(edit.plcolor(i),'foregroundcolor','w')
    end

    edit.plsize(i)=uicontrol('parent',edit.plpanel(i),'style','edit',...
        'units','normalized','position',[.09 .15 .15 .2],'background','w',...
        'callback','spikeHound(''gUpdateGraphics'')');
    set(edit.plsize(i),'string',num2str(get(edit.pl(i),'linewidth')),'userdata',...
        get(edit.pl(i),'linewidth'))

    %Markers
    uicontrol('parent',edit.plpanel(i),'style','text','backgroundcolor',...
        get(edit.axpanel,'backgroundcolor'),'units','normalized','position',...
        [.4 .87 .15 .1],'string','Marker')
    edit.mkstyle(i)=uicontrol('parent',edit.plpanel(i),'style','popupmenu','backgroundcolor',...
        'w','units','normalized','callback','spikeHound(''gUpdateGraphics'')','position',...
        [.37 .75 .2 .1],'string',strvcat('+','o','*','.','x','square','diamond','v','^','>','<','pentagram','hexagram','none'),...
        'horizontalalignment','right');
    switch get(edit.pl(i),'marker')
        case '+'; set(edit.mkstyle(i),'value',1)
        case 'o'; set(edit.mkstyle(i),'value',2)
        case '*'; set(edit.mkstyle(i),'value',3)
        case '.'; set(edit.mkstyle(i),'value',4)
        case 'x'; set(edit.mkstyle(i),'value',5)
        case 'square'; set(edit.mkstyle(i),'value',6)
        case 'diamond'; set(edit.mkstyle(i),'value',7)
        case 'v'; set(edit.mkstyle(i),'value',8)
        case '^'; set(edit.mkstyle(i),'value',9)
        case '>'; set(edit.mkstyle(i),'value',10)
        case '<'; set(edit.mkstyle(i),'value',11)
        case 'pentagram'; set(edit.mkstyle(i),'value',12)
        case 'hexagram'; set(edit.mkstyle(i),'value',13)
        case 'none'; set(edit.mkstyle(i),'value',14)
    end
    
    edit.mkfcolor(i)=uicontrol('parent',edit.plpanel(i),'style','pushbutton',...
        'units','normalized','position',[.36 .4 .1 .2],'string','Face',...
        'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
    set(edit.pl(i),'markerfacecolor',get(edit.pl(i),'color'))
    set(edit.mkfcolor(i),'backgroundcolor',get(edit.pl(i),'markerfacecolor'))
    if sum(get(edit.mkfcolor(i),'backgroundcolor'))/3>.5
        set(edit.mkfcolor(i),'foregroundcolor','k')
    else
        set(edit.mkfcolor(i),'foregroundcolor','w')
    end

    edit.mkecolor(i)=uicontrol('parent',edit.plpanel(i),'style','pushbutton',...
        'units','normalized','position',[.48 .4 .1 .2],'string','Edge',...
        'callback','spikeHound(''ColorSelector''); spikeHound(''gUpdateGraphics'')');
    set(edit.pl(i),'markeredgecolor',get(edit.pl(i),'color'))
    set(edit.mkecolor(i),'backgroundcolor',get(edit.pl(i),'markeredgecolor'))

    if sum(get(edit.mkecolor(i),'backgroundcolor'))/3>.5
        set(edit.mkecolor(i),'foregroundcolor','k')
    else
        set(edit.mkecolor(i),'foregroundcolor','w')
    end

    edit.mksize(i)=uicontrol('parent',edit.plpanel(i),'style','edit',...
        'units','normalized','position',[.39 .15 .15 .2],'backgroundcolor','w',...
        'callback','spikeHound(''gUpdateGraphics'')');
    set(edit.mksize,'string',num2str(get(edit.pl(i),'markersize')),...
        'userdata',get(edit.pl(i),'markersize'))    
    
    %visibility controls
    edit.plvisible(i)=uicontrol('parent',edit.plpanel(i),'style','checkbox','string','Visible',...
        'units','normalized','position',[.75 .7 .15 .2],'backgroundcolor',get(edit.plpanel(i),'backgroundcolor'),...
        'callback','spikeHound(''gUpdateGraphics'')');
    if strcmpi(get(edit.pl(i),'visible'),'on')
        set(edit.plvisible(i),'value',1)
    end
    
    edit.plMoveUp(i)=uicontrol('parent',edit.plpanel(i),'style','pushbutton','string','Move Forward',...
        'units','normalized','position',[.7 .4 .25 .2],'backgroundcolor',get(edit.plpanel(i),'backgroundcolor'),...
        'callback','spikeHound(''gMoveTraceUp'')');
    edit.plMoveBack(i)=uicontrol('parent',edit.plpanel(i),'style','pushbutton','string','Move Back',...
        'units','normalized','position',[.7 .15 .25 .2],'backgroundcolor',get(edit.plpanel(i),'backgroundcolor'),...
        'callback','spikeHound(''gMoveTraceBack'')');
   
end

if length(edit.pl)~=0
    set(edit.plpanel(1),'visible','on')
%list of plots (drop down) outside of the panel to select from
    edit.PlotSelect=uicontrol('style','popupmenu','backgroundcolor',...
        'w','units','normalized','position',[.07 -.02 .2 .1],'string',' ');
    for i=1:length(edit.pl)
        s{i}=['Plot ',num2str(i)];
    end
    set(edit.PlotSelect,'string',s,'callback','spikeHound(''gPropSelectTrace'')')
end

set(gcf,'userdata',edit)

function gPropSelectTrace
edit=get(gcbf,'userdata');
set(edit.plpanel,'visible','off')
set(edit.plpanel(get(gcbo,'value')),'visible','on')

%Select a color for an object
function ColorSelector
edit=get(gcbf,'userdata');
c=uisetcolor('Select a Color');
if length(c)>1
    set(gcbo,'backgroundcolor',c)
    if mean(c)>0.5; set(gcbo,'foregroundcolor','k'); 
    else; set(gcbo,'foregroundcolor','w'); end
end

function gMoveTraceUp
edit=get(gcbf,'userdata');

thisplot=edit.pl(find(edit.plMoveUp==gcbo));
order=get(edit.ax,'children');
ind=find(thisplot==order);
if ind==1; return; end

order(ind)=order(ind-1);
order(ind-1)=thisplot;
set(edit.ax,'children',order)

function gMoveTraceBack
edit=get(gcbf,'userdata');

thisplot=edit.pl(find(edit.plMoveBack==gcbo));
order=get(edit.ax,'children');
ind=find(thisplot==order);
if ind==length(order); return; end

order(ind)=order(ind+1);
order(ind+1)=thisplot;
set(edit.ax,'children',order)

%Sweep through all interface elements and load them into the axis
function gUpdateGraphics
edit=get(gcbf,'userdata');

%Check to see if the length of plots in the axis is different than the
%length of plots in the edit struct

title(edit.ax,get(edit.axTitle,'string'))
xlabel(edit.ax,get(edit.axXLabel,'string'))
ylabel(edit.ax,get(edit.axYLabel,'string'))
set(edit.ax,'color',get(edit.axBackColor,'backgroundcolor'))
set(edit.ax,'xcolor',get(edit.axXColor,'backgroundcolor'))
set(edit.ax,'ycolor',get(edit.axYColor,'backgroundcolor'))

if get(edit.axXLog,'value'); set(edit.ax,'xscale','Log')
else set(edit.ax,'xscale','linear'); end

if get(edit.axYLog,'value'); set(edit.ax,'yscale','Log')
else set(edit.ax,'yscale','linear'); end

xlim1=str2double(get(edit.axXlow,'string'));
if isnan(xlim1); set(edit.axXlow,'string',num2str(get(edit.axXlow,'userdata')));
else set(edit.axXlow,'userdata',xlim1); end
xlim1=get(edit.axXlow,'userdata');

xlim2=str2double(get(edit.axXhigh,'string'));
if isnan(xlim2); set(edit.axXhigh,'string',num2str(get(edit.axXhigh,'userdata')));
else set(edit.axXhigh,'userdata',xlim2); end
xlim2=get(edit.axXhigh,'userdata');

ylim1=str2double(get(edit.axYlow,'string'));
if isnan(ylim1); set(edit.axYlow,'string',num2str(get(edit.axYlow,'userdata')));
else set(edit.axYlow,'userdata',ylim1); end
ylim1=get(edit.axYlow,'userdata');

ylim2=str2double(get(edit.axYhigh,'string'));
if isnan(ylim2); set(edit.axYhigh,'string',num2str(get(edit.axYhigh,'userdata')));
else set(edit.axYhigh,'userdata',ylim2); end
ylim2=get(edit.axYhigh,'userdata');

xlim=sort([xlim1 xlim2]);
ylim=sort([ylim1 ylim2]);
set(edit.axXlow,'userdata',xlim(1),'string',num2str(xlim(1)))
set(edit.axXhigh,'userdata',xlim(2),'string',num2str(xlim(2)))
set(edit.axYlow,'userdata',ylim(1),'string',num2str(ylim(1)))
set(edit.axYhigh,'userdata',ylim(2),'string',num2str(ylim(2)))


if get(edit.axXLimAuto,'value')
    set(edit.ax,'xlimmode','auto')
    set([edit.axXlow edit.axXhigh],'enable','off')
else
    set(edit.ax,'xlim',xlim)
    set([edit.axXlow edit.axXhigh],'enable','on')
end

if get(edit.axYLimAuto,'value')
    set(edit.ax,'ylimmode','auto')
    set([edit.axYlow edit.axYhigh],'enable','off')
else
    set(edit.ax,'ylim',ylim)
    set([edit.axYlow edit.axYhigh],'enable','on')
end


xlim=get(edit.ax,'xlim');
ylim=get(edit.ax,'ylim');
set(edit.axXlow,'userdata',xlim(1),'string',xlim(1))
set(edit.axXhigh,'userdata',xlim(2),'string',xlim(2))
set(edit.axYlow,'userdata',ylim(1),'string',ylim(1))
set(edit.axYhigh,'userdata',ylim(2),'string',ylim(2))

if get(edit.axXgrid,'value'); set(edit.ax,'xgrid','on'); 
else set(edit.ax,'xgrid','off'); end
if get(edit.axYgrid,'value'); set(edit.ax,'ygrid','on'); 
else set(edit.ax,'ygrid','off'); end




for i=1:length(edit.pl)
    set(edit.pl(i),'linestyle',popupstr(edit.plstyle(i)))
    set(edit.pl(i),'color',get(edit.plcolor(i),'backgroundcolor'))
    
    lw=str2double(get(edit.plsize(i),'string'));
    if isnan(lw); set(edit.plsize(i),'string',num2str(get(edit.plsize(i),'userdata'))) 
    else set(edit.plsize(i),'userdata',lw); end
    lw=get(edit.plsize(i),'userdata');
    set(edit.pl(i),'linewidth',lw)
    
    set(edit.pl(i),'marker',popupstr(edit.mkstyle(i)))
    set(edit.pl(i),'markerfacecolor',get(edit.mkfcolor(i),'backgroundcolor'))
    set(edit.pl(i),'markeredgecolor',get(edit.mkecolor(i),'backgroundcolor'))
    
    mw=str2double(get(edit.mksize(i),'string'));
    if isnan(mw); set(edit.mksize(i),'string',num2str(get(edit.mksize(i),'userdata'))) 
    else set(edit.mksize(i),'userdata',mw); end
    mw=get(edit.mksize(i),'userdata');
    set(edit.pl(i),'markersize',mw)
    
    if get(edit.plvisible(i),'value'); set(edit.pl(i),'visible','on');
    else set(edit.pl(i),'visible','off'); end
    
end

%Function Generator Handling

function FCNAddElement(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');

ElementList=get(gui.FunctionGenerator,'userdata');
gTemp=uipanel('parent',gui.FunctionGenerator,'units','normalized','position',[0.23 0.03 0.34 0.96],...
    'tag','FCNGenElement','backgroundcolor',[.8 .5 .8]);
ElementList=[ElementList,gTemp];

pres=get(gui.FCNtypeList,'string');
pren=0;

gTs=get(gui.FCNtype,'string');
gTs=gTs{get(gui.FCNtype,'value')};
FCN.Name=uicontrol('parent',gTemp','style','text','backgroundcolor',get(gTemp,'Backgroundcolor'),'units','normalized',...
    'string','','position',[0.01 0.85 0.3 0.15],'horizontalalignment','left','fontweight','bold','userdata',gTs);
uicontrol('parent',gTemp','style','text','backgroundcolor',get(gTemp,'Backgroundcolor'),'units','normalized',...
    'string','Amplitude:','position',[0.01 0.68 0.24 0.15],'horizontalalignment','left');
FCN.Amp=uicontrol('parent',gTemp','style','edit','backgroundcolor','w','units','normalized',...
    'string','1','position',[0.25 0.68 0.18 0.15],'userdata',1,'callback',{@FCNupdateSig,gTemp});
uicontrol('parent',gTemp','style','text','backgroundcolor',get(gTemp,'Backgroundcolor'),'units','normalized',...
    'string','Freq (Hz):','position',[0.01 0.48 0.24 0.15],'horizontalalignment','left');
FCN.Frequency=uicontrol('parent',gTemp','style','edit','backgroundcolor','w','units','normalized',...
    'string','500','position',[0.25 0.48 0.18 0.15],'userdata',500,'callback',{@FCNupdateSig,gTemp});

uicontrol('parent',gTemp','style','text','backgroundcolor',get(gTemp,'Backgroundcolor'),'units','normalized',...
    'string','Duty Cycle (%):','position',[0.5 0.68 0.3 0.15],'horizontalalignment','left');
FCN.Duty=uicontrol('parent',gTemp','style','edit','backgroundcolor','w','units','normalized',...
    'string','50','position',[0.8 0.68 0.18 0.15],'userdata',50,'callback',{@FCNupdateSig,gTemp});
uicontrol('parent',gTemp','style','text','backgroundcolor',get(gTemp,'Backgroundcolor'),'units','normalized',...
    'string','Phase (deg):','position',[0.5 0.48 0.3 0.15],'horizontalalignment','left');
FCN.Phase=uicontrol('parent',gTemp','style','edit','backgroundcolor','w','units','normalized',...
    'string','0','position',[0.8 0.48 0.18 0.15],'userdata',0,'callback',{@FCNupdateSig,gTemp});

FCN.ExampleAx=axes('parent',gTemp,'position',[0.01 0.01 0.98 0.45],'xticklabel',[],'yticklabel',[],'visible','off');
FCN.ExamplePl=line(nan,nan,'parent',FCN.ExampleAx);

switch get(gui.FCNtype,'value')
    case 1 %Sine
        for i=1:length(pres); pren=pren+length(strfind(pres{i},'Sine')); end
        pres{end+1}=['Sine ',num2str(pren+1)];
        set(FCN.Duty,'enable','off')
    case 2 %Square
        for i=1:length(pres); pren=pren+length(strfind(pres{i},'Square')); end
        pres{end+1}=['Square ',num2str(pren+1)];
    case 3 %Triangle
        for i=1:length(pres); pren=pren+length(strfind(pres{i},'Triangle')); end
        pres{end+1}=['Triangle ',num2str(pren+1)];
    case 4 %White Noise
        for i=1:length(pres); pren=pren+length(strfind(pres{i},'Noise')); end
        pres{end+1}=['Noise ',num2str(pren+1)];
        set([FCN.Phase, FCN.Duty, FCN.Frequency],'enable','off','string','','userdata',[])
    case 5 %DC Offset
        for i=1:length(pres); pren=pren+length(strfind(pres{i},'DC Offset')); end
        pres{end+1}=['DC Offset ',num2str(pren+1)];
        set([FCN.Phase, FCN.Duty, FCN.Frequency],'enable','off','string','','userdata',[])
end

set(gTemp,'tag',pres{end},'userdata',FCN)
set(gui.FCNtypeList,'string',pres,'value',length(pres))
set(gui.FunctionGenerator,'userdata',ElementList)
set(FCN.Name,'string',pres{end})

FCNupdateSig([],'',gTemp)

%Switch active panel to front
function FCNSelectElement(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
s=get(gui.FCNtypeList,'string');
if isempty(s); return; end

val=get(gui.FCNtypeList,'value');
set(ElementList,'visible','off')
set(ElementList(val),'visible','on')

%Remove a Single Function Generator Signal Element
function FCNRemoveElement(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
s=get(gui.FCNtypeList,'string');
if isempty(s); return; end

val=get(gui.FCNtypeList,'value');

panel=findobj('type','uipanel','tag',s{val});
ElementList(ElementList==panel)=[];
delete(panel)
set(gui.FunctionGenerator,'userdata',ElementList)
s(val)=[];
if val>length(s); val=length(s); end

set(gui.FCNtypeList,'string',s,'value',val)
if isempty(s)
    set(gui.FCNtypeGen,'value',0)
    FCNGenGo([],'')
    set(gui.FCNOutPl,'xdata',nan,'ydata',nan);
end
FCNSelectElement([],'')
FCNUpdateFullPlot([],'')

%Remove all function Generator Signal Elements
function FCNClearAll(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
s=get(gui.FCNtypeList,'string');
if isempty(s); return; end

delete(ElementList)
set(gui.FunctionGenerator,'userdata',[])
set(gui.FCNtypeList,'string',{},'value',0)
set(gui.FCNOutPl,'ydata',nan,'xdata',nan)
set(gui.FCNtypeGen,'value',0)
FCNGenGo([],'')

function FCNChangeSrate(obj,event,aoactive)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
if isempty(ElementList); set(gui.FCNOutPl,'xdata',nan,'ydata',nan); return; end

for i=ElementList
    FCNupdateSig([],'',i);
end

if aoactive
    FCNGenGo([],'')
end

%Update an Example Sig for a given frame
function FCNupdateSig(obj,event,panel)
gui=get(findobj('tag','gSS07'),'userdata');
FCN=get(panel,'userdata');

%Exception Handling for Interface Elements
if ishandle(obj)
switch obj
    case FCN.Amp
        s=str2double(get(obj,'string'));
        if isnan(s); set(obj,'string',num2str(get(obj,'userdata'))); end
        set(obj,'userdata',str2double(get(obj,'string')))
    case FCN.Frequency
        s=str2double(get(obj,'string'));
        if isnan(s)|s<=0; set(obj,'string',num2str(get(obj,'userdata'))); end
        set(obj,'userdata',str2double(get(obj,'string')))
    case FCN.Duty
        s=str2double(get(obj,'string'));
        if isnan(s)|s<0|s>100; set(obj,'string',num2str(get(obj,'userdata'))); end
        set(obj,'userdata',str2double(get(obj,'string')))
    case FCN.Phase
        s=str2double(get(obj,'string'));
        if isnan(s); set(obj,'string',num2str(get(obj,'userdata'))); end
        set(obj,'userdata',str2double(get(obj,'string')))
end
end

amp=get(FCN.Amp,'userdata');
freq=get(FCN.Frequency,'userdata');
duty=get(FCN.Duty,'userdata');
phase=get(FCN.Phase,'userdata');

gT=daqhwinfo(gui.ao);
%If dealing w/ USB 6009 or 8
if (strcmpi(gT.DeviceName,'USB-6009')|strcmpi(gT.DeviceName,'USB-6008'))&obj==FCN.Amp
    putsample(gui.ao,[1 1]*amp)
    return
end


params.amp=amp; params.freq=freq; params.duty=duty; params.phase=phase; params.type=get(FCN.Name,'userdata');

sRate=gui.ao.samplerate;
time=linspace(0,499/sRate,500);
sig=zeros(1,500);

switch params.type
    case 'Sine'
            %Create One period of the signal
            wt=(0:(2*pi/sRate):2*pi)*freq;
            sig=amp*sin(wt+phase*pi/180);
            sig=sig(wt<6*pi);
            
            time=(0:(length(sig)-1))/length(sig);
    case 'Square'
            wt=(0:(2*pi/sRate):2*pi)*freq;
            sig=amp*square(wt+phase*pi/180,duty);
            sig=sig(wt<6*pi);
            
            time=(0:(length(sig)-1))/length(sig);
    case 'Triangle'
            wt=(0:(2*pi/sRate):2*pi)*freq;
            sig=amp*sawtooth(wt+phase*pi/180,duty/100);
            sig=sig(wt<6*pi);
            
            time=(0:(length(sig)-1))/length(sig);
    case 'White Noise'
            sig=amp*rand(1,200);
            
            time=(0:(length(sig)-1))/length(sig);
    case 'DC Offset'
            sig=zeros(1,200)+amp;
            
            time=(0:(length(sig)-1))/length(sig);
end

newtime=zeros(length(time)*2-1,1);
newtime(1:2:end)=time;
newtime(2:2:end)=time(2:end);

newsig=zeros(length(sig)*2-1,1);
newsig(1:2:end)=sig;
newsig(2:2:end)=sig(1:(end-1));
set(FCN.ExamplePl,'xdata',newtime,'ydata',newsig,'userdata',params);
set(FCN.ExampleAx,'userdata',sig)
FCNUpdateFullPlot([],'')


%Draw full sig in output example display
function FCNUpdateFullPlot(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
if isempty(ElementList); return; end
    
lengths=[];
for i=1:length(ElementList)
    FCN=get(ElementList(i),'userdata');
    sig{i}=get(FCN.ExampleAx,'userdata');
    lengths=[lengths,length(sig{i})];
end

longest=lengths(find(lengths==max(lengths),1,'first'));

for i=1:length(sig)
    longsig=repmat(sig{i},[1 ceil(longest/lengths(i))]);
    longsig=longsig(1:longest);
    sig{i}=longsig;
end



if isempty(sig); return; end

outsig=sig{1};
for i=2:length(sig)
    outsig=outsig+sig{i};
end

time=1:length(outsig);

newtime=zeros(length(time)*2-1,1);
newtime(1:2:end)=time;
newtime(2:2:end)=time(2:end);

newsig=zeros(length(outsig)*2-1,1);
newsig(1:2:end)=outsig;
newsig(2:2:end)=outsig(1:(end-1));

if max(newsig)>max(gui.ao.channel(1).outputrange)|min(newsig)<min(gui.ao.channel(1).outputrange)
    set(gui.FCNTitle,'String','OUTPUT CLIPPING')
else
    set(gui.FCNTitle,'String','')
end

s=get(gui.OutputRange,'string');
nowOR=str2num(s{get(gui.OutputRange,'value')});
newsig(newsig>max(nowOR))=max(nowOR);
newsig(newsig<min(nowOR))=min(nowOR);

set(gui.FCNOutPl,'ydata',newsig,'xdata',newtime)
set(gui.FCNOutAx,'xlim',[1 length(outsig)])

%Handle Initializing the Function Generation Output and Cleanup afterwards
function FCNGenGo(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
ElementList=get(gui.FunctionGenerator,'userdata');
if isempty(ElementList); set(gui.FCNtypeGen,'value',0); end

if get(gui.FCNtypeGen,'value')==0;
    stop(gui.ao)
    gui.ao.TimerFcn='';
    delete(gui.ao.channel)
    gT=daqhwinfo(gui.ao);
    addchannel(gui.ao,gT.ChannelIDs(1:2));
    s=get(gui.OutputRange,'string');
    gui.ao.channel.outputrange=str2num(s{get(gui.OutputRange,'value')});
    return
end

if isempty(ElementList); return; end
if strcmpi(gui.ao.running,'on'); return; end

gui.ao.repeatoutput=0;
delete(gui.ao.channel)
gT=daqhwinfo(gui.ao);

addchannel(gui.ao,gT.ChannelIDs(1:2));
s=get(gui.OutputRange,'string');
gui.ao.channel.outputrange=str2num(s{get(gui.OutputRange,'value')});

%Branch if dealing with NIDAQ USB-6008/9
if strcmpi(gT.DeviceName,'USB-6009')|strcmpi(gT.DeviceName,'USB-6008')
    set(gui.FCNtypeGen,'value',0)
    return
end

uddobj = daqgetfield(gui.ao,'uddobject');

gui.ao.SamplesOutputFcnCount=round(0.2*gui.ao.samplerate);
gui.ao.SamplesOutputFcn={@FCNOutputTimer,gui.ao.samplerate,gui.ao.SamplesOutputFcnCount,gui.ao.channel(1).outputrange,gui,uddobj};
gui.ao.StopFcn=@FCNStopFcn;
gui.ao.TimerFcn={@FCNClipReset,gui.FCNTitle};
gui.ao.TimerPeriod=0.5;

t0=0;
t1=gui.ao.SamplesOutputFcnCount*4-1;
n=gui.ao.SamplesOutputFcnCount*4;

wt=linspace(t0,t1,n)*2*pi/gui.ao.samplerate;
sig=zeros(1,length(wt));

for i=1:length(ElementList)
    FCN=get(ElementList(i),'userdata');
    params=get(FCN.ExamplePl,'userdata');
    switch params.type
        case 'Sine'
            wt=linspace(t0,t1,n)*2*pi*params.freq/gui.ao.samplerate;
            sig=sig+params.amp*sin(wt+params.phase*pi/180);
        case 'Square'
            wt=linspace(t0,t1,n)*2*pi*params.freq/gui.ao.samplerate;
            basesig=sign(diff(sawtooth(wt+params.phase*pi/180,params.duty/100)));
            sig=sig+params.amp*[basesig,basesig(end)];
        case 'Triangle'
            wt=linspace(t0,t1,n)*2*pi*params.freq/gui.ao.samplerate;
            sig=sig+params.amp*sawtooth(wt+params.phase*pi/180,params.duty/100);
        case 'White Noise'
            wt=linspace(t0,t1,n)*2*pi/gui.ao.samplerate;
            sig=sig+params.amp*2*(rand(1,length(wt))-0.5);
        case 'DC Offset'
            sig=sig+params.amp;
    end
end

if max(sig)>max(gui.ao.channel(1).outputrange)|min(sig)<min(gui.ao.channel(1).outputrange)
    set(gui.FCNTitle,'String','OUTPUT CLIPPING')
end
    
%keep index in the AO userdata

set(gui.FCNOutAx,'userdata',t1)

%add it to the AO
putdata(uddobj,[sig(:) sig(:)]); 

%Run
start(gui.ao)

function FCNClipReset(obj,event,txt)
set(txt,'String','');


%This function handles the output of 
function FCNOutputTimer(obj,event,sRate,SOFC,COR,gui,uddobj)
ElementList=get(gui.FunctionGenerator,'userdata');
if isempty(ElementList); return; end

    t0=get(gui.FCNOutAx,'userdata');
    t1=t0+SOFC;
    n=SOFC;
    wt=linspace(t0,t1,n+1)*2*pi/sRate;
    wt=wt(2:end);
    sig=zeros(1,length(wt));
    
for i=1:length(ElementList)
    FCN=get(ElementList(i),'userdata');
    params=get(FCN.ExamplePl,'userdata');
  
    switch params.type
        case 'Sine'
            wt=linspace(t0,t1,n+1)*2*pi*params.freq/sRate;
            wt=wt(2:end);
            sig=sig+params.amp*sin(wt+params.phase*pi/180);
        case 'Square'
            wt=linspace(t0,t1,n+1)*2*pi*params.freq/sRate;
            wt=wt(2:end);
            basesig=sign(diff(sawtooth(wt+params.phase*pi/180,params.duty/100)));
            sig=sig+params.amp*[basesig,basesig(end)];
        case 'Triangle'
            wt=linspace(t0,t1,n+1)*2*pi*params.freq/sRate;
            wt=wt(2:end);
            sig=sig+params.amp*sawtooth(wt+params.phase*pi/180,params.duty/100);
        case 'White Noise'
            wt=linspace(t0,t1,n+1)*2*pi/sRate;
            wt=wt(2:end);
            sig=sig+params.amp*2*(rand(1,length(wt))-0.5);
        case 'DC Offset'
            sig=sig+params.amp;
    end

end

if max(sig)>max(COR)|min(sig)<min(COR)
    set(gui.FCNTitle,'String','OUTPUT CLIPPING')
end

set(gui.FCNOutAx,'userdata',t1)
putdata(uddobj,[sig(:) sig(:)]); 


function FCNStopFcn(obj,event)
gui=get(findobj('tag','gSS07'),'userdata');
set(gui.FCNtypeGen,'value',0)


%EOF