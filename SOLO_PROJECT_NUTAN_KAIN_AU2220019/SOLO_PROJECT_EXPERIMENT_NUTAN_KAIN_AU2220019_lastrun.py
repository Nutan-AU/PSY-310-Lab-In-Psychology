#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.1.5),
    on December 03, 2024, at 20:38
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.1.5'
expName = 'SOLO_PROJECT_EXPERIMENT_NUTAN_KAIN_AU2220019'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1366, 768]
_loggingLevel = logging.getLevel('warning')
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # override logging level
    _loggingLevel = logging.getLevel(
        prefs.piloting['pilotLoggingLevel']
    )

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Lenovo\\Documents\\MON 2024-25\\FULL_TERM\\PSY-310_LAB_IN_PSYCHOLOGY\\0_SOLO_PROJECT_NUTAN_KAIN_AU2220019_310\\SOLO_PROJECT_EXPERIMENT_NUTAN_KAIN_AU2220019_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(_loggingLevel)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=_loggingLevel)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_resp_4') is None:
        # initialise key_resp_4
        key_resp_4 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_4',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "guidelines" ---
    text = visual.TextStim(win=win, name='text',
        text="hello, \nIn this task you will be shown an image which you shall later on try to find within a set of images. With the help of the mouse, you shall click on the correct image. \n\nIf you have read and understood the guidelines and would like to participate, press SPACEBAR. if you'd like to withdraw, please press ESC.\n\nThank You",
        font='Monospace',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "find_1" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='In the upcoming task, you shall try to find this image and click on it with the help of the mouse to go forward. If you have looked at the picture and would like to begin, please press the SPACEBAR. ',
        font='Monospace',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    find_image_1 = visual.ImageStim(
        win=win,
        name='find_image_1', 
        image='C:/Users/Lenovo/Documents/MON 2024-25/FULL_TERM/PSY-310_LAB_IN_PSYCHOLOGY/0_SOLO_PROJECT_NUTAN_KAIN_AU2220019_310/subtle_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.05), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "baseline_sa" ---
    subtle_anger = visual.ImageStim(
        win=win,
        name='subtle_anger', 
        image='subtle_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    n_1 = visual.ImageStim(
        win=win,
        name='n_1', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    n_2 = visual.ImageStim(
        win=win,
        name='n_2', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    n_3 = visual.ImageStim(
        win=win,
        name='n_3', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    n_4 = visual.ImageStim(
        win=win,
        name='n_4', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    n_5 = visual.ImageStim(
        win=win,
        name='n_5', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    n_6 = visual.ImageStim(
        win=win,
        name='n_6', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    mouse_resp = event.Mouse(win=win)
    x, y = [None, None]
    mouse_resp.mouseClock = core.Clock()
    fix_1 = visual.ShapeStim(
        win=win, name='fix_1', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    
    # --- Initialize components for Routine "main_sa" ---
    subtle_anger_main = visual.ImageStim(
        win=win,
        name='subtle_anger_main', 
        image='subtle_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    mouse_resp_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_resp_4.mouseClock = core.Clock()
    d_1 = visual.ImageStim(
        win=win,
        name='d_1', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    d_2 = visual.ImageStim(
        win=win,
        name='d_2', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    d_3 = visual.ImageStim(
        win=win,
        name='d_3', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    s_1 = visual.ImageStim(
        win=win,
        name='s_1', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    s_2 = visual.ImageStim(
        win=win,
        name='s_2', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    s_3 = visual.ImageStim(
        win=win,
        name='s_3', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    fix_2 = visual.ShapeStim(
        win=win, name='fix_2', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    
    # --- Initialize components for Routine "find_3" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='In the upcoming task, you shall try to find this image and click on it with the help of the mouse to go forward. If you have looked at the picture and would like to begin, please press the SPACEBAR. ',
        font='Monospace',
        pos=(0, 0.2), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard(deviceName='key_resp_4')
    find_image__ = visual.ImageStim(
        win=win,
        name='find_image__', 
        image='C:/Users/Lenovo/Documents/MON 2024-25/FULL_TERM/PSY-310_LAB_IN_PSYCHOLOGY/0_SOLO_PROJECT_NUTAN_KAIN_AU2220019_310/clear_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.05), size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "baseline_ca" ---
    clear_anger = visual.ImageStim(
        win=win,
        name='clear_anger', 
        image='clear_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    n_7 = visual.ImageStim(
        win=win,
        name='n_7', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    n_8 = visual.ImageStim(
        win=win,
        name='n_8', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    n_9 = visual.ImageStim(
        win=win,
        name='n_9', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    n_10 = visual.ImageStim(
        win=win,
        name='n_10', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    n_11 = visual.ImageStim(
        win=win,
        name='n_11', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    n_12 = visual.ImageStim(
        win=win,
        name='n_12', 
        image='neutral.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    mouse_resp_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_resp_3.mouseClock = core.Clock()
    fix_3 = visual.ShapeStim(
        win=win, name='fix_3', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    
    # --- Initialize components for Routine "main_ca" ---
    clear_anger_main = visual.ImageStim(
        win=win,
        name='clear_anger_main', 
        image='clear_anger.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    mouse_resp_5 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_resp_5.mouseClock = core.Clock()
    d_4 = visual.ImageStim(
        win=win,
        name='d_4', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    d_5 = visual.ImageStim(
        win=win,
        name='d_5', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    d_6 = visual.ImageStim(
        win=win,
        name='d_6', 
        image='disgust.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    s_4 = visual.ImageStim(
        win=win,
        name='s_4', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    s_5 = visual.ImageStim(
        win=win,
        name='s_5', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    s_6 = visual.ImageStim(
        win=win,
        name='s_6', 
        image='sad.jpg', mask=None, anchor='center',
        ori=0.0, pos=[0,0], size=(0.2, 0.2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-7.0)
    fix_4 = visual.ShapeStim(
        win=win, name='fix_4', vertices='cross',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, 0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-8.0, interpolate=True)
    
    # --- Initialize components for Routine "thank_you" ---
    thanks = visual.TextStim(win=win, name='thanks',
        text='The experiment has ended.\nThank You for your Participation. ',
        font='Monospace',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "guidelines" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('guidelines.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # keep track of which components have finished
    guidelinesComponents = [text, key_resp]
    for thisComponent in guidelinesComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "guidelines" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in guidelinesComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "guidelines" ---
    for thisComponent in guidelinesComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('guidelines.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "guidelines" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "find_1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('find_1.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    find_1Components = [text_2, find_image_1, key_resp_2]
    for thisComponent in find_1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "find_1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *find_image_1* updates
        
        # if find_image_1 is starting this frame...
        if find_image_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            find_image_1.frameNStart = frameN  # exact frame index
            find_image_1.tStart = t  # local t and not account for scr refresh
            find_image_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(find_image_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'find_image_1.started')
            # update status
            find_image_1.status = STARTED
            find_image_1.setAutoDraw(True)
        
        # if find_image_1 is active this frame...
        if find_image_1.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in find_1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "find_1" ---
    for thisComponent in find_1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('find_1.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "find_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_1 = data.TrialHandler(nReps=50.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_1')
    thisExp.addLoop(loop_1)  # add the loop to the experiment
    thisLoop_1 = loop_1.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
    if thisLoop_1 != None:
        for paramName in thisLoop_1:
            globals()[paramName] = thisLoop_1[paramName]
    
    for thisLoop_1 in loop_1:
        currentLoop = loop_1
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
        if thisLoop_1 != None:
            for paramName in thisLoop_1:
                globals()[paramName] = thisLoop_1[paramName]
        
        # --- Prepare to start Routine "baseline_sa" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('baseline_sa.started', globalClock.getTime(format='float'))
        subtle_anger.setPos((random()-0.5, random()-0.5))
        n_1.setPos((random()-0.5, random()-0.5))
        n_2.setPos((random()-0.5, random()-0.5))
        n_3.setPos((random()-0.5, random()-0.5))
        n_4.setPos((random()-0.5, random()-0.5))
        n_5.setPos((random()-0.5, random()-0.5))
        n_6.setPos((random()-0.5, random()-0.5))
        # setup some python lists for storing info about the mouse_resp
        mouse_resp.x = []
        mouse_resp.y = []
        mouse_resp.leftButton = []
        mouse_resp.midButton = []
        mouse_resp.rightButton = []
        mouse_resp.time = []
        mouse_resp.corr = []
        mouse_resp.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        baseline_saComponents = [subtle_anger, n_1, n_2, n_3, n_4, n_5, n_6, mouse_resp, fix_1]
        for thisComponent in baseline_saComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "baseline_sa" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *subtle_anger* updates
            
            # if subtle_anger is starting this frame...
            if subtle_anger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                subtle_anger.frameNStart = frameN  # exact frame index
                subtle_anger.tStart = t  # local t and not account for scr refresh
                subtle_anger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(subtle_anger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'subtle_anger.started')
                # update status
                subtle_anger.status = STARTED
                subtle_anger.setAutoDraw(True)
            
            # if subtle_anger is active this frame...
            if subtle_anger.status == STARTED:
                # update params
                pass
            
            # *n_1* updates
            
            # if n_1 is starting this frame...
            if n_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_1.frameNStart = frameN  # exact frame index
                n_1.tStart = t  # local t and not account for scr refresh
                n_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_1.started')
                # update status
                n_1.status = STARTED
                n_1.setAutoDraw(True)
            
            # if n_1 is active this frame...
            if n_1.status == STARTED:
                # update params
                pass
            
            # *n_2* updates
            
            # if n_2 is starting this frame...
            if n_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_2.frameNStart = frameN  # exact frame index
                n_2.tStart = t  # local t and not account for scr refresh
                n_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_2.started')
                # update status
                n_2.status = STARTED
                n_2.setAutoDraw(True)
            
            # if n_2 is active this frame...
            if n_2.status == STARTED:
                # update params
                pass
            
            # *n_3* updates
            
            # if n_3 is starting this frame...
            if n_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_3.frameNStart = frameN  # exact frame index
                n_3.tStart = t  # local t and not account for scr refresh
                n_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_3.started')
                # update status
                n_3.status = STARTED
                n_3.setAutoDraw(True)
            
            # if n_3 is active this frame...
            if n_3.status == STARTED:
                # update params
                pass
            
            # *n_4* updates
            
            # if n_4 is starting this frame...
            if n_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_4.frameNStart = frameN  # exact frame index
                n_4.tStart = t  # local t and not account for scr refresh
                n_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_4.started')
                # update status
                n_4.status = STARTED
                n_4.setAutoDraw(True)
            
            # if n_4 is active this frame...
            if n_4.status == STARTED:
                # update params
                pass
            
            # *n_5* updates
            
            # if n_5 is starting this frame...
            if n_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_5.frameNStart = frameN  # exact frame index
                n_5.tStart = t  # local t and not account for scr refresh
                n_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_5.started')
                # update status
                n_5.status = STARTED
                n_5.setAutoDraw(True)
            
            # if n_5 is active this frame...
            if n_5.status == STARTED:
                # update params
                pass
            
            # *n_6* updates
            
            # if n_6 is starting this frame...
            if n_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_6.frameNStart = frameN  # exact frame index
                n_6.tStart = t  # local t and not account for scr refresh
                n_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_6.started')
                # update status
                n_6.status = STARTED
                n_6.setAutoDraw(True)
            
            # if n_6 is active this frame...
            if n_6.status == STARTED:
                # update params
                pass
            # *mouse_resp* updates
            
            # if mouse_resp is starting this frame...
            if mouse_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_resp.frameNStart = frameN  # exact frame index
                mouse_resp.tStart = t  # local t and not account for scr refresh
                mouse_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_resp.started', t)
                # update status
                mouse_resp.status = STARTED
                mouse_resp.mouseClock.reset()
                prevButtonState = mouse_resp.getPressed()  # if button is down already this ISN'T a new click
            if mouse_resp.status == STARTED:  # only update if started and not finished!
                buttons = mouse_resp.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(subtle_anger, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_resp):
                                gotValidClick = True
                                mouse_resp.clicked_name.append(obj.name)
                        # check whether click was in correct object
                        if gotValidClick:
                            _corr = 0
                            _corrAns = environmenttools.getFromNames(subtle_anger, namespace=locals())
                            for obj in _corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse_resp):
                                    _corr = 1
                            mouse_resp.corr.append(_corr)
                        x, y = mouse_resp.getPos()
                        mouse_resp.x.append(x)
                        mouse_resp.y.append(y)
                        buttons = mouse_resp.getPressed()
                        mouse_resp.leftButton.append(buttons[0])
                        mouse_resp.midButton.append(buttons[1])
                        mouse_resp.rightButton.append(buttons[2])
                        mouse_resp.time.append(mouse_resp.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *fix_1* updates
            
            # if fix_1 is starting this frame...
            if fix_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_1.frameNStart = frameN  # exact frame index
                fix_1.tStart = t  # local t and not account for scr refresh
                fix_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_1.started')
                # update status
                fix_1.status = STARTED
                fix_1.setAutoDraw(True)
            
            # if fix_1 is active this frame...
            if fix_1.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in baseline_saComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "baseline_sa" ---
        for thisComponent in baseline_saComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('baseline_sa.stopped', globalClock.getTime(format='float'))
        # store data for loop_1 (TrialHandler)
        loop_1.addData('mouse_resp.x', mouse_resp.x)
        loop_1.addData('mouse_resp.y', mouse_resp.y)
        loop_1.addData('mouse_resp.leftButton', mouse_resp.leftButton)
        loop_1.addData('mouse_resp.midButton', mouse_resp.midButton)
        loop_1.addData('mouse_resp.rightButton', mouse_resp.rightButton)
        loop_1.addData('mouse_resp.time', mouse_resp.time)
        loop_1.addData('mouse_resp.corr', mouse_resp.corr)
        loop_1.addData('mouse_resp.clicked_name', mouse_resp.clicked_name)
        # the Routine "baseline_sa" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "main_sa" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('main_sa.started', globalClock.getTime(format='float'))
        subtle_anger_main.setPos((random()-0.5, random()-0.5))
        # setup some python lists for storing info about the mouse_resp_4
        mouse_resp_4.x = []
        mouse_resp_4.y = []
        mouse_resp_4.leftButton = []
        mouse_resp_4.midButton = []
        mouse_resp_4.rightButton = []
        mouse_resp_4.time = []
        mouse_resp_4.corr = []
        mouse_resp_4.clicked_name = []
        gotValidClick = False  # until a click is received
        d_1.setPos((random()-0.5, random()-0.5))
        d_2.setPos((random()-0.5, random()-0.5))
        d_3.setPos((random()-0.5, random()-0.5))
        s_1.setPos((random()-0.5, random()-0.5))
        s_2.setPos((random()-0.5, random()-0.5))
        s_3.setPos((random()-0.5, random()-0.5))
        # keep track of which components have finished
        main_saComponents = [subtle_anger_main, mouse_resp_4, d_1, d_2, d_3, s_1, s_2, s_3, fix_2]
        for thisComponent in main_saComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_sa" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *subtle_anger_main* updates
            
            # if subtle_anger_main is starting this frame...
            if subtle_anger_main.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                subtle_anger_main.frameNStart = frameN  # exact frame index
                subtle_anger_main.tStart = t  # local t and not account for scr refresh
                subtle_anger_main.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(subtle_anger_main, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'subtle_anger_main.started')
                # update status
                subtle_anger_main.status = STARTED
                subtle_anger_main.setAutoDraw(True)
            
            # if subtle_anger_main is active this frame...
            if subtle_anger_main.status == STARTED:
                # update params
                pass
            # *mouse_resp_4* updates
            
            # if mouse_resp_4 is starting this frame...
            if mouse_resp_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_resp_4.frameNStart = frameN  # exact frame index
                mouse_resp_4.tStart = t  # local t and not account for scr refresh
                mouse_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_resp_4.started', t)
                # update status
                mouse_resp_4.status = STARTED
                mouse_resp_4.mouseClock.reset()
                prevButtonState = mouse_resp_4.getPressed()  # if button is down already this ISN'T a new click
            if mouse_resp_4.status == STARTED:  # only update if started and not finished!
                buttons = mouse_resp_4.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(subtle_anger_main, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_resp_4):
                                gotValidClick = True
                                mouse_resp_4.clicked_name.append(obj.name)
                        # check whether click was in correct object
                        if gotValidClick:
                            _corr = 0
                            _corrAns = environmenttools.getFromNames(subtle_anger_main, namespace=locals())
                            for obj in _corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse_resp_4):
                                    _corr = 1
                            mouse_resp_4.corr.append(_corr)
                        x, y = mouse_resp_4.getPos()
                        mouse_resp_4.x.append(x)
                        mouse_resp_4.y.append(y)
                        buttons = mouse_resp_4.getPressed()
                        mouse_resp_4.leftButton.append(buttons[0])
                        mouse_resp_4.midButton.append(buttons[1])
                        mouse_resp_4.rightButton.append(buttons[2])
                        mouse_resp_4.time.append(mouse_resp_4.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *d_1* updates
            
            # if d_1 is starting this frame...
            if d_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_1.frameNStart = frameN  # exact frame index
                d_1.tStart = t  # local t and not account for scr refresh
                d_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_1.started')
                # update status
                d_1.status = STARTED
                d_1.setAutoDraw(True)
            
            # if d_1 is active this frame...
            if d_1.status == STARTED:
                # update params
                pass
            
            # *d_2* updates
            
            # if d_2 is starting this frame...
            if d_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_2.frameNStart = frameN  # exact frame index
                d_2.tStart = t  # local t and not account for scr refresh
                d_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_2.started')
                # update status
                d_2.status = STARTED
                d_2.setAutoDraw(True)
            
            # if d_2 is active this frame...
            if d_2.status == STARTED:
                # update params
                pass
            
            # *d_3* updates
            
            # if d_3 is starting this frame...
            if d_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_3.frameNStart = frameN  # exact frame index
                d_3.tStart = t  # local t and not account for scr refresh
                d_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_3.started')
                # update status
                d_3.status = STARTED
                d_3.setAutoDraw(True)
            
            # if d_3 is active this frame...
            if d_3.status == STARTED:
                # update params
                pass
            
            # *s_1* updates
            
            # if s_1 is starting this frame...
            if s_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_1.frameNStart = frameN  # exact frame index
                s_1.tStart = t  # local t and not account for scr refresh
                s_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_1, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_1.started')
                # update status
                s_1.status = STARTED
                s_1.setAutoDraw(True)
            
            # if s_1 is active this frame...
            if s_1.status == STARTED:
                # update params
                pass
            
            # *s_2* updates
            
            # if s_2 is starting this frame...
            if s_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_2.frameNStart = frameN  # exact frame index
                s_2.tStart = t  # local t and not account for scr refresh
                s_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_2.started')
                # update status
                s_2.status = STARTED
                s_2.setAutoDraw(True)
            
            # if s_2 is active this frame...
            if s_2.status == STARTED:
                # update params
                pass
            
            # *s_3* updates
            
            # if s_3 is starting this frame...
            if s_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_3.frameNStart = frameN  # exact frame index
                s_3.tStart = t  # local t and not account for scr refresh
                s_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_3.started')
                # update status
                s_3.status = STARTED
                s_3.setAutoDraw(True)
            
            # if s_3 is active this frame...
            if s_3.status == STARTED:
                # update params
                pass
            
            # *fix_2* updates
            
            # if fix_2 is starting this frame...
            if fix_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_2.frameNStart = frameN  # exact frame index
                fix_2.tStart = t  # local t and not account for scr refresh
                fix_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_2.started')
                # update status
                fix_2.status = STARTED
                fix_2.setAutoDraw(True)
            
            # if fix_2 is active this frame...
            if fix_2.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_saComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_sa" ---
        for thisComponent in main_saComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('main_sa.stopped', globalClock.getTime(format='float'))
        # store data for loop_1 (TrialHandler)
        loop_1.addData('mouse_resp_4.x', mouse_resp_4.x)
        loop_1.addData('mouse_resp_4.y', mouse_resp_4.y)
        loop_1.addData('mouse_resp_4.leftButton', mouse_resp_4.leftButton)
        loop_1.addData('mouse_resp_4.midButton', mouse_resp_4.midButton)
        loop_1.addData('mouse_resp_4.rightButton', mouse_resp_4.rightButton)
        loop_1.addData('mouse_resp_4.time', mouse_resp_4.time)
        loop_1.addData('mouse_resp_4.corr', mouse_resp_4.corr)
        loop_1.addData('mouse_resp_4.clicked_name', mouse_resp_4.clicked_name)
        # the Routine "main_sa" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 50.0 repeats of 'loop_1'
    
    
    # --- Prepare to start Routine "find_3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('find_3.started', globalClock.getTime(format='float'))
    # create starting attributes for key_resp_4
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    find_3Components = [text_4, key_resp_4, find_image__]
    for thisComponent in find_3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "find_3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *find_image__* updates
        
        # if find_image__ is starting this frame...
        if find_image__.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            find_image__.frameNStart = frameN  # exact frame index
            find_image__.tStart = t  # local t and not account for scr refresh
            find_image__.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(find_image__, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'find_image__.started')
            # update status
            find_image__.status = STARTED
            find_image__.setAutoDraw(True)
        
        # if find_image__ is active this frame...
        if find_image__.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in find_3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "find_3" ---
    for thisComponent in find_3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('find_3.stopped', globalClock.getTime(format='float'))
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "find_3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    loop_2 = data.TrialHandler(nReps=50.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='loop_2')
    thisExp.addLoop(loop_2)  # add the loop to the experiment
    thisLoop_2 = loop_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLoop_2.rgb)
    if thisLoop_2 != None:
        for paramName in thisLoop_2:
            globals()[paramName] = thisLoop_2[paramName]
    
    for thisLoop_2 in loop_2:
        currentLoop = loop_2
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_2.rgb)
        if thisLoop_2 != None:
            for paramName in thisLoop_2:
                globals()[paramName] = thisLoop_2[paramName]
        
        # --- Prepare to start Routine "baseline_ca" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('baseline_ca.started', globalClock.getTime(format='float'))
        clear_anger.setPos((random()-0.5, random()-0.5))
        n_7.setPos((random()-0.5, random()-0.5))
        n_8.setPos((random()-0.5, random()-0.5))
        n_9.setPos((random()-0.5, random()-0.5))
        n_10.setPos((random()-0.5, random()-0.5))
        n_11.setPos((random()-0.5, random()-0.5))
        n_12.setPos((random()-0.5, random()-0.5))
        # setup some python lists for storing info about the mouse_resp_3
        mouse_resp_3.x = []
        mouse_resp_3.y = []
        mouse_resp_3.leftButton = []
        mouse_resp_3.midButton = []
        mouse_resp_3.rightButton = []
        mouse_resp_3.time = []
        mouse_resp_3.corr = []
        mouse_resp_3.clicked_name = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        baseline_caComponents = [clear_anger, n_7, n_8, n_9, n_10, n_11, n_12, mouse_resp_3, fix_3]
        for thisComponent in baseline_caComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "baseline_ca" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *clear_anger* updates
            
            # if clear_anger is starting this frame...
            if clear_anger.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                clear_anger.frameNStart = frameN  # exact frame index
                clear_anger.tStart = t  # local t and not account for scr refresh
                clear_anger.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(clear_anger, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'clear_anger.started')
                # update status
                clear_anger.status = STARTED
                clear_anger.setAutoDraw(True)
            
            # if clear_anger is active this frame...
            if clear_anger.status == STARTED:
                # update params
                pass
            
            # *n_7* updates
            
            # if n_7 is starting this frame...
            if n_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_7.frameNStart = frameN  # exact frame index
                n_7.tStart = t  # local t and not account for scr refresh
                n_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_7, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_7.started')
                # update status
                n_7.status = STARTED
                n_7.setAutoDraw(True)
            
            # if n_7 is active this frame...
            if n_7.status == STARTED:
                # update params
                pass
            
            # *n_8* updates
            
            # if n_8 is starting this frame...
            if n_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_8.frameNStart = frameN  # exact frame index
                n_8.tStart = t  # local t and not account for scr refresh
                n_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_8.started')
                # update status
                n_8.status = STARTED
                n_8.setAutoDraw(True)
            
            # if n_8 is active this frame...
            if n_8.status == STARTED:
                # update params
                pass
            
            # *n_9* updates
            
            # if n_9 is starting this frame...
            if n_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_9.frameNStart = frameN  # exact frame index
                n_9.tStart = t  # local t and not account for scr refresh
                n_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_9.started')
                # update status
                n_9.status = STARTED
                n_9.setAutoDraw(True)
            
            # if n_9 is active this frame...
            if n_9.status == STARTED:
                # update params
                pass
            
            # *n_10* updates
            
            # if n_10 is starting this frame...
            if n_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_10.frameNStart = frameN  # exact frame index
                n_10.tStart = t  # local t and not account for scr refresh
                n_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_10.started')
                # update status
                n_10.status = STARTED
                n_10.setAutoDraw(True)
            
            # if n_10 is active this frame...
            if n_10.status == STARTED:
                # update params
                pass
            
            # *n_11* updates
            
            # if n_11 is starting this frame...
            if n_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_11.frameNStart = frameN  # exact frame index
                n_11.tStart = t  # local t and not account for scr refresh
                n_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_11.started')
                # update status
                n_11.status = STARTED
                n_11.setAutoDraw(True)
            
            # if n_11 is active this frame...
            if n_11.status == STARTED:
                # update params
                pass
            
            # *n_12* updates
            
            # if n_12 is starting this frame...
            if n_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                n_12.frameNStart = frameN  # exact frame index
                n_12.tStart = t  # local t and not account for scr refresh
                n_12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(n_12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'n_12.started')
                # update status
                n_12.status = STARTED
                n_12.setAutoDraw(True)
            
            # if n_12 is active this frame...
            if n_12.status == STARTED:
                # update params
                pass
            # *mouse_resp_3* updates
            
            # if mouse_resp_3 is starting this frame...
            if mouse_resp_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_resp_3.frameNStart = frameN  # exact frame index
                mouse_resp_3.tStart = t  # local t and not account for scr refresh
                mouse_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_resp_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_resp_3.started', t)
                # update status
                mouse_resp_3.status = STARTED
                mouse_resp_3.mouseClock.reset()
                prevButtonState = mouse_resp_3.getPressed()  # if button is down already this ISN'T a new click
            if mouse_resp_3.status == STARTED:  # only update if started and not finished!
                buttons = mouse_resp_3.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(clear_anger, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_resp_3):
                                gotValidClick = True
                                mouse_resp_3.clicked_name.append(obj.name)
                        # check whether click was in correct object
                        if gotValidClick:
                            _corr = 0
                            _corrAns = environmenttools.getFromNames(clear_anger, namespace=locals())
                            for obj in _corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse_resp_3):
                                    _corr = 1
                            mouse_resp_3.corr.append(_corr)
                        x, y = mouse_resp_3.getPos()
                        mouse_resp_3.x.append(x)
                        mouse_resp_3.y.append(y)
                        buttons = mouse_resp_3.getPressed()
                        mouse_resp_3.leftButton.append(buttons[0])
                        mouse_resp_3.midButton.append(buttons[1])
                        mouse_resp_3.rightButton.append(buttons[2])
                        mouse_resp_3.time.append(mouse_resp_3.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *fix_3* updates
            
            # if fix_3 is starting this frame...
            if fix_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_3.frameNStart = frameN  # exact frame index
                fix_3.tStart = t  # local t and not account for scr refresh
                fix_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_3.started')
                # update status
                fix_3.status = STARTED
                fix_3.setAutoDraw(True)
            
            # if fix_3 is active this frame...
            if fix_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in baseline_caComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "baseline_ca" ---
        for thisComponent in baseline_caComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('baseline_ca.stopped', globalClock.getTime(format='float'))
        # store data for loop_2 (TrialHandler)
        loop_2.addData('mouse_resp_3.x', mouse_resp_3.x)
        loop_2.addData('mouse_resp_3.y', mouse_resp_3.y)
        loop_2.addData('mouse_resp_3.leftButton', mouse_resp_3.leftButton)
        loop_2.addData('mouse_resp_3.midButton', mouse_resp_3.midButton)
        loop_2.addData('mouse_resp_3.rightButton', mouse_resp_3.rightButton)
        loop_2.addData('mouse_resp_3.time', mouse_resp_3.time)
        loop_2.addData('mouse_resp_3.corr', mouse_resp_3.corr)
        loop_2.addData('mouse_resp_3.clicked_name', mouse_resp_3.clicked_name)
        # the Routine "baseline_ca" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "main_ca" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('main_ca.started', globalClock.getTime(format='float'))
        clear_anger_main.setPos((random()-0.5, random()-0.5))
        # setup some python lists for storing info about the mouse_resp_5
        mouse_resp_5.x = []
        mouse_resp_5.y = []
        mouse_resp_5.leftButton = []
        mouse_resp_5.midButton = []
        mouse_resp_5.rightButton = []
        mouse_resp_5.time = []
        mouse_resp_5.corr = []
        mouse_resp_5.clicked_name = []
        gotValidClick = False  # until a click is received
        d_4.setPos((random()-0.5, random()-0.5))
        d_5.setPos((random()-0.5, random()-0.5))
        d_6.setPos((random()-0.5, random()-0.5))
        s_4.setPos((random()-0.5, random()-0.5))
        s_5.setPos((random()-0.5, random()-0.5))
        s_6.setPos((random()-0.5, random()-0.5))
        # keep track of which components have finished
        main_caComponents = [clear_anger_main, mouse_resp_5, d_4, d_5, d_6, s_4, s_5, s_6, fix_4]
        for thisComponent in main_caComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "main_ca" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *clear_anger_main* updates
            
            # if clear_anger_main is starting this frame...
            if clear_anger_main.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                clear_anger_main.frameNStart = frameN  # exact frame index
                clear_anger_main.tStart = t  # local t and not account for scr refresh
                clear_anger_main.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(clear_anger_main, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'clear_anger_main.started')
                # update status
                clear_anger_main.status = STARTED
                clear_anger_main.setAutoDraw(True)
            
            # if clear_anger_main is active this frame...
            if clear_anger_main.status == STARTED:
                # update params
                pass
            # *mouse_resp_5* updates
            
            # if mouse_resp_5 is starting this frame...
            if mouse_resp_5.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                mouse_resp_5.frameNStart = frameN  # exact frame index
                mouse_resp_5.tStart = t  # local t and not account for scr refresh
                mouse_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(mouse_resp_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('mouse_resp_5.started', t)
                # update status
                mouse_resp_5.status = STARTED
                mouse_resp_5.mouseClock.reset()
                prevButtonState = mouse_resp_5.getPressed()  # if button is down already this ISN'T a new click
            if mouse_resp_5.status == STARTED:  # only update if started and not finished!
                buttons = mouse_resp_5.getPressed()
                if buttons != prevButtonState:  # button state changed?
                    prevButtonState = buttons
                    if sum(buttons) > 0:  # state changed to a new click
                        # check if the mouse was inside our 'clickable' objects
                        gotValidClick = False
                        clickableList = environmenttools.getFromNames(clear_anger_main, namespace=locals())
                        for obj in clickableList:
                            # is this object clicked on?
                            if obj.contains(mouse_resp_5):
                                gotValidClick = True
                                mouse_resp_5.clicked_name.append(obj.name)
                        # check whether click was in correct object
                        if gotValidClick:
                            _corr = 0
                            _corrAns = environmenttools.getFromNames(clear_anger_main, namespace=locals())
                            for obj in _corrAns:
                                # is this object clicked on?
                                if obj.contains(mouse_resp_5):
                                    _corr = 1
                            mouse_resp_5.corr.append(_corr)
                        x, y = mouse_resp_5.getPos()
                        mouse_resp_5.x.append(x)
                        mouse_resp_5.y.append(y)
                        buttons = mouse_resp_5.getPressed()
                        mouse_resp_5.leftButton.append(buttons[0])
                        mouse_resp_5.midButton.append(buttons[1])
                        mouse_resp_5.rightButton.append(buttons[2])
                        mouse_resp_5.time.append(mouse_resp_5.mouseClock.getTime())
                        
                        continueRoutine = False  # end routine on response
            
            # *d_4* updates
            
            # if d_4 is starting this frame...
            if d_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_4.frameNStart = frameN  # exact frame index
                d_4.tStart = t  # local t and not account for scr refresh
                d_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_4.started')
                # update status
                d_4.status = STARTED
                d_4.setAutoDraw(True)
            
            # if d_4 is active this frame...
            if d_4.status == STARTED:
                # update params
                pass
            
            # *d_5* updates
            
            # if d_5 is starting this frame...
            if d_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_5.frameNStart = frameN  # exact frame index
                d_5.tStart = t  # local t and not account for scr refresh
                d_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_5.started')
                # update status
                d_5.status = STARTED
                d_5.setAutoDraw(True)
            
            # if d_5 is active this frame...
            if d_5.status == STARTED:
                # update params
                pass
            
            # *d_6* updates
            
            # if d_6 is starting this frame...
            if d_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                d_6.frameNStart = frameN  # exact frame index
                d_6.tStart = t  # local t and not account for scr refresh
                d_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(d_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'd_6.started')
                # update status
                d_6.status = STARTED
                d_6.setAutoDraw(True)
            
            # if d_6 is active this frame...
            if d_6.status == STARTED:
                # update params
                pass
            
            # *s_4* updates
            
            # if s_4 is starting this frame...
            if s_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_4.frameNStart = frameN  # exact frame index
                s_4.tStart = t  # local t and not account for scr refresh
                s_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_4.started')
                # update status
                s_4.status = STARTED
                s_4.setAutoDraw(True)
            
            # if s_4 is active this frame...
            if s_4.status == STARTED:
                # update params
                pass
            
            # *s_5* updates
            
            # if s_5 is starting this frame...
            if s_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_5.frameNStart = frameN  # exact frame index
                s_5.tStart = t  # local t and not account for scr refresh
                s_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_5, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_5.started')
                # update status
                s_5.status = STARTED
                s_5.setAutoDraw(True)
            
            # if s_5 is active this frame...
            if s_5.status == STARTED:
                # update params
                pass
            
            # *s_6* updates
            
            # if s_6 is starting this frame...
            if s_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                s_6.frameNStart = frameN  # exact frame index
                s_6.tStart = t  # local t and not account for scr refresh
                s_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(s_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 's_6.started')
                # update status
                s_6.status = STARTED
                s_6.setAutoDraw(True)
            
            # if s_6 is active this frame...
            if s_6.status == STARTED:
                # update params
                pass
            
            # *fix_4* updates
            
            # if fix_4 is starting this frame...
            if fix_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fix_4.frameNStart = frameN  # exact frame index
                fix_4.tStart = t  # local t and not account for scr refresh
                fix_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fix_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fix_4.started')
                # update status
                fix_4.status = STARTED
                fix_4.setAutoDraw(True)
            
            # if fix_4 is active this frame...
            if fix_4.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in main_caComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "main_ca" ---
        for thisComponent in main_caComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('main_ca.stopped', globalClock.getTime(format='float'))
        # store data for loop_2 (TrialHandler)
        loop_2.addData('mouse_resp_5.x', mouse_resp_5.x)
        loop_2.addData('mouse_resp_5.y', mouse_resp_5.y)
        loop_2.addData('mouse_resp_5.leftButton', mouse_resp_5.leftButton)
        loop_2.addData('mouse_resp_5.midButton', mouse_resp_5.midButton)
        loop_2.addData('mouse_resp_5.rightButton', mouse_resp_5.rightButton)
        loop_2.addData('mouse_resp_5.time', mouse_resp_5.time)
        loop_2.addData('mouse_resp_5.corr', mouse_resp_5.corr)
        loop_2.addData('mouse_resp_5.clicked_name', mouse_resp_5.clicked_name)
        # the Routine "main_ca" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 50.0 repeats of 'loop_2'
    
    
    # --- Prepare to start Routine "thank_you" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thank_you.started', globalClock.getTime(format='float'))
    # keep track of which components have finished
    thank_youComponents = [thanks]
    for thisComponent in thank_youComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thank_you" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks* updates
        
        # if thanks is starting this frame...
        if thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks.frameNStart = frameN  # exact frame index
            thanks.tStart = t  # local t and not account for scr refresh
            thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks.started')
            # update status
            thanks.status = STARTED
            thanks.setAutoDraw(True)
        
        # if thanks is active this frame...
        if thanks.status == STARTED:
            # update params
            pass
        
        # if thanks is stopping this frame...
        if thanks.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanks.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                thanks.tStop = t  # not accounting for scr refresh
                thanks.tStopRefresh = tThisFlipGlobal  # on global time
                thanks.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanks.stopped')
                # update status
                thanks.status = FINISHED
                thanks.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thank_youComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thank_you" ---
    for thisComponent in thank_youComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thank_you.stopped', globalClock.getTime(format='float'))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    # shut down eyetracker, if there is one
    if deviceManager.getDevice('eyetracker') is not None:
        deviceManager.removeDevice('eyetracker')
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
