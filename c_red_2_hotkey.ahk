#Requires AutoHotkey v2.0-beta
#SingleInstance Force

; init variables
button1Pos := Map()
button2Pos := Map()
counter := 1
suffix := ""  ; add suffix
configFile := A_ScriptDir "\button_config.ini"

; load saved config
LoadConfig()

; show debug info
^!q::ShowDebugInfo()

; Ctrl+Alt+S - use mouse center button to set the position of button#1
^!s:: {
    global button1Pos, counter, suffix

    ; Prompt user to input suffix
    userInput := InputBox("Input suffix:", "Suffix Setting", , suffix)
    if (userInput.Result != "OK") {  ; User cancelled
        ToolTip("Cancelled")
        SetTimer(ClearToolTip, -2000)
        return
    }
    suffix := userInput.Value  ; Save user input suffix

    ; msg box for user
    ToolTip("Use mouse center button to set the position of button#1...")

    ; wait for user click
    KeyWait("MButton", "D")
    Sleep(100)

    ; get current pos of mouse
    MouseGetPos(&xpos, &ypos, &windowID)

    if (windowID) {
        button1Pos["x"] := xpos
        button1Pos["y"] := ypos
        button1Pos["winID"] := windowID
        button1Pos["winTitle"] := WinGetTitle("ahk_id " windowID)
        button1Pos["winClass"] := WinGetClass("ahk_id " windowID)
        ; counter := 1  ; reset counter
        SaveConfig()

        ToolTip(
            "Button#1 position recorded:`n"
            "X=" xpos ", Y=" ypos "`n"
            "Window=" button1Pos["winTitle"] "`n"
            "Suffix=" suffix
        )
        SetTimer(ClearToolTip, -5000)
    } else {
        ToolTip("Cannot obtain window ID, please retry")
        SetTimer(ClearToolTip, -3000)
    }
}

; Ctrl+Alt+D - Set button#2 position
^!d::
{
    global button2Pos
    
    ToolTip("Use mouse center button to set the position of button#2...")

    KeyWait("MButton", "D")
    Sleep(100) 
    
    MouseGetPos(&xpos, &ypos, &windowID)
    
    if (windowID) {
        ; save info to config file
        button2Pos["x"] := xpos
        button2Pos["y"] := ypos
        button2Pos["winID"] := windowID
        
        winTitle := WinGetTitle("ahk_id " windowID)
        winClass := WinGetClass("ahk_id " windowID)
        button2Pos["winTitle"] := winTitle
        button2Pos["winClass"] := winClass
        
        SaveConfig()
        
        ToolTip("Button#2 position recorded: X=" xpos ", Y=" ypos "`nWindow: " winTitle)
        SetTimer(ClearToolTip, -3000) 
    } else {
        ToolTip("Cannot obtain window ID, please retry")
        SetTimer(ClearToolTip, -3000)
    }
}

; Ctrl+Alt+Z - Run button#1 callbacks
^!z::
{
    global button1Pos, counter, suffix
    
    ; check if button pos set
    if (!button1Pos.Has("x")) {
        ToolTip("Button#1 position not set. Please press Ctrl+Alt+S first to set position for Button#1")
        SetTimer(ClearToolTip, -3000)
        return
    }
    
    targetWindow := "ahk_id " button1Pos["winID"]
    if (!WinExist(targetWindow)) {
        targetWindow := "ahk_class " button1Pos["winClass"]
        if (!WinExist(targetWindow)) {
            ToolTip("Cannot find window#1")
            SetTimer(ClearToolTip, -3000)
            return
        }
    }
    
    WinActivate(targetWindow)
    WinWaitActive(targetWindow, , 2)
    
    Click(button1Pos["x"], button1Pos["y"])
    
    Sleep(500)
    
    Loop 4
        Send("{Tab}")
		Sleep(200)
    
    Send("2000")
	Sleep(200)
    
    ; Press Tab key 9 times
    Loop 9
        Send("{Tab}")
		Sleep(100)
    
	; Press Enter key
    Send("{Enter}")
	Sleep(250)
	
    ; Input incrementing filename with suffix
    if (suffix != "")
        Send(counter "_" suffix)
    else
        Send(counter)
    
    ; Increment counter
    counter++
    
    ; Save counter to config file
    SaveConfig()
    
    ; Press Tab key once more
    Send("{Tab}")
	Sleep(200)
    
    ; Input "h"
    Send("h")
	Sleep(200)
    
}

; Ctrl+Alt+X - Execute button#2 operation
^!x::
{
    global button2Pos
    
    ; Check if button position is set
    if (!button2Pos.Has("x")) {
        ToolTip("Button#2 position not set. Please press Ctrl+Alt+D first to set position.")
        SetTimer(ClearToolTip, -3000)
        return
    }
    
    ; Try to activate target window
    targetWindow := "ahk_id " button2Pos["winID"]
    if (!WinExist(targetWindow)) {
        ; If ID fails, try using window title and class
        targetWindow := "ahk_class " button2Pos["winClass"]
        if (!WinExist(targetWindow)) {
            ToolTip("Cannot find button#2's target window.")
            SetTimer(ClearToolTip, -3000)
            return
        }
    }
    
    WinActivate(targetWindow)
    WinWaitActive(targetWindow, , 2)
    
    ; Click button#2
    Click(button2Pos["x"], button2Pos["y"])
}

; Function to clear tooltip
ClearToolTip() {
    ToolTip()
}

; Function to save configuration to INI file
SaveConfig() {
    global button1Pos, button2Pos, counter, configFile, suffix
    
    try {
        ; Save button#1 info
        if (button1Pos.Has("x")) {
            IniWrite(button1Pos["x"], configFile, "Button1", "X")
            IniWrite(button1Pos["y"], configFile, "Button1", "Y")
            IniWrite(button1Pos["winID"], configFile, "Button1", "WinID")
            IniWrite(button1Pos["winTitle"], configFile, "Button1", "WinTitle")
            IniWrite(button1Pos["winClass"], configFile, "Button1", "WinClass")
            
            ; Show save confirmation
            ToolTip("Button#1 info saved to: " configFile)
            SetTimer(ClearToolTip, -2000)
        }
        
        ; Save button#2 info
        if (button2Pos.Has("x")) {
            IniWrite(button2Pos["x"], configFile, "Button2", "X")
            IniWrite(button2Pos["y"], configFile, "Button2", "Y")
            IniWrite(button2Pos["winID"], configFile, "Button2", "WinID")
            IniWrite(button2Pos["winTitle"], configFile, "Button2", "WinTitle")
            IniWrite(button2Pos["winClass"], configFile, "Button2", "WinClass")
        }
        
        ; Save counter and suffix
        ; IniWrite(counter, configFile, "General", "Counter")
        IniWrite(suffix, configFile, "General", "Suffix")
    } catch Error as e {
        ToolTip("Error saving configuration: " e.Message "`nAttempted to save to: " configFile)
        SetTimer(ClearToolTip, -5000)
    }
}

; Function to load configuration from INI file
LoadConfig() {
    global button1Pos, button2Pos, counter, configFile, suffix
    
    if (FileExist(configFile)) {
        try {
            ; Load button#1 info
            if (IniRead(configFile, "Button1", "X", "ERROR") != "ERROR") {
                button1Pos["x"] := Integer(IniRead(configFile, "Button1", "X", 0))
                button1Pos["y"] := Integer(IniRead(configFile, "Button1", "Y", 0))
                button1Pos["winID"] := IniRead(configFile, "Button1", "WinID", 0)
                button1Pos["winTitle"] := IniRead(configFile, "Button1", "WinTitle", "")
                button1Pos["winClass"] := IniRead(configFile, "Button1", "WinClass", "")
                
                ; Show load confirmation
                ToolTip("Button#1 config loaded: X=" button1Pos["x"] ", Y=" button1Pos["y"])
                SetTimer(ClearToolTip, -2000)
            }
            
            ; Load button#2 info
            if (IniRead(configFile, "Button2", "X", "ERROR") != "ERROR") {
                button2Pos["x"] := Integer(IniRead(configFile, "Button2", "X", 0))
                button2Pos["y"] := Integer(IniRead(configFile, "Button2", "Y", 0))
                button2Pos["winID"] := IniRead(configFile, "Button2", "WinID", 0)
                button2Pos["winTitle"] := IniRead(configFile, "Button2", "WinTitle", "")
                button2Pos["winClass"] := IniRead(configFile, "Button2", "WinClass", "")
            }
            
            ; Load suffix
            ; counter := Integer(IniRead(configFile, "General", "Counter", 1))
            suffix := IniRead(configFile, "General", "Suffix", "")
        } catch Error as e {
            ToolTip("Error loading configuration: " e.Message)
            SetTimer(ClearToolTip, -5000)
        }
    } else {
        ToolTip("Configuration file not found, using default values")
        SetTimer(ClearToolTip, -3000)
    }
}

; Add debug function
ShowDebugInfo() {
    global button1Pos, button2Pos, counter, configFile, suffix
    
    ; Build debug info text
    debugText := "===== Debug Information =====`n"
    debugText .= "Config file: " configFile "`n"
    debugText .= "Config file exists: " (FileExist(configFile) ? "Yes" : "No") "`n`n"
    
    debugText .= "Button#1 status: " (button1Pos.Has("x") ? "Set" : "Not set") "`n"
    if (button1Pos.Has("x")) {
        debugText .= "X: " button1Pos["x"] ", Y: " button1Pos["y"] "`n"
        debugText .= "Window ID: " button1Pos["winID"] "`n"
        debugText .= "Window Title: " button1Pos["winTitle"] "`n"
        debugText .= "Window Class: " button1Pos["winClass"] "`n`n"
    }
    
    debugText .= "Button#2 status: " (button2Pos.Has("x") ? "Set" : "Not set") "`n"
    if (button2Pos.Has("x")) {
        debugText .= "X: " button2Pos["x"] ", Y: " button2Pos["y"] "`n"
        debugText .= "Window ID: " button2Pos["winID"] "`n"
        debugText .= "Window Title: " button2Pos["winTitle"] "`n"
        debugText .= "Window Class: " button2Pos["winClass"] "`n`n"
    }
    
    debugText .= "Counter: " counter "`n"
    debugText .= "Suffix: " suffix "`n"
    
    ; Display info
    MsgBox(debugText, "Script Debug Information", "T10")
}