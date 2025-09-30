#Requires AutoHotkey v2.0-beta
#SingleInstance Force

; 初始化变量
button2Pos := Map()
configFile := A_ScriptDir "\button_config.ini"

; 主函数 - 启动时自动执行
Main()
{
    ; 显示正在启动的通知
    ToolTip("正在读取配置并执行点击...")
    
    ; 加载配置
    if (!LoadConfig()) {
        MsgBox("无法加载配置文件或按钮2未设置。", "错误", "Icon!")
        ExitApp
    }
    
    ; 尝试执行点击
    if (!ClickButton2()) {
        MsgBox("执行点击操作失败。", "错误", "Icon!")
        ExitApp
    }
    
    ; 显示完成通知
    ToolTip("完成！")
    Sleep(500)
    ToolTip()
    
    ; 退出脚本
    ExitApp
}

; 从INI文件加载按钮2的配置
LoadConfig() {
    global button2Pos, configFile
    
    ; 检查配置文件是否存在
    if (!FileExist(configFile)) {
        ToolTip("配置文件不存在: " configFile)
        Sleep(2000)
        ToolTip()
        return false
    }
    
    try {
        ; 检查按钮2是否已设置
        if (IniRead(configFile, "Button2", "X", "ERROR") == "ERROR") {
            ToolTip("配置文件中没有按钮2的信息")
            Sleep(2000)
            ToolTip()
            return false
        }
        
        ; 加载按钮2信息
        button2Pos["x"] := Integer(IniRead(configFile, "Button2", "X", 0))
        button2Pos["y"] := Integer(IniRead(configFile, "Button2", "Y", 0))
        button2Pos["winID"] := IniRead(configFile, "Button2", "WinID", 0)
        button2Pos["winTitle"] := IniRead(configFile, "Button2", "WinTitle", "")
        button2Pos["winClass"] := IniRead(configFile, "Button2", "WinClass", "")
        
        return true
    } catch Error as e {
        ToolTip("加载配置时出错: " e.Message)
        Sleep(2000)
        ToolTip()
        return false
    }
}

; 执行按钮2的点击操作
ClickButton2() {
    global button2Pos
    
    ; 检查按钮位置是否已设置
    if (!button2Pos.Has("x")) {
        return false
    }
    
    try {
        ; 尝试激活目标窗口
        targetWindow := "ahk_id " button2Pos["winID"]
        if (!WinExist(targetWindow)) {
            ; 如果ID失败，尝试使用窗口标题和类名
            targetWindow := "ahk_class " button2Pos["winClass"]
            if (!WinExist(targetWindow)) {
                ToolTip("未找到按钮2的目标窗口。")
                Sleep(2000)
                ToolTip()
                return false
            }
        }
        
        WinActivate(targetWindow)
        WinWaitActive(targetWindow, , 2)
        
        ; 点击按钮2
        Click(button2Pos["x"], button2Pos["y"])
        
        return true
    } catch Error as e {
        ToolTip("点击过程中出错: " e.Message)
        Sleep(2000)
        ToolTip()
        return false
    }
}

; 启动主函数
Main()