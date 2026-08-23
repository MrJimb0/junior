-- Junior Workbench — double-click the .app built from this to open the workbench.
--
-- An app rather than a script file, for one reason. Terminal runs a .command or a bare
-- executable by TYPING its path into a fresh interactive shell, so whatever that shell
-- asks on startup answers itself with the keystrokes. An oh-my-zsh update prompt ate
-- the leading "/" of the path and left "zsh: no such file or directory: Users/...".
-- AppleScript's `do shell script` goes through /bin/sh -c, which reads no startup
-- files, so there is nothing there to ask a question.
--
-- The APP owns the server's lifetime: it stays open while the workbench is up, and
-- quitting it stops the server. That job used to belong to a dialog sitting in the run
-- handler, and the dialog is what made the app unclickable. An applet blocked in
-- `display dialog` processes no further launches, and while ANY instance is running
-- macOS answers a double-click by activating it instead of running the script — so the
-- second click did nothing at all. Worse, when the server died under it the applet sat
-- there forever and every later click went the same way. Reported exactly that way,
-- twice.
--
-- Junior is resolved from THIS bundle's own folder, never from $PATH. A stray `junior`
-- earlier on someone's PATH would seal and run different code against this checkout's
-- data, and the failure would look like Junior misbehaving rather than like the wrong
-- Junior. Not finding one is a dialog saying how to install, not a silent fallback.
--
-- The source lives here rather than beside the app, so the repository root holds one
-- obvious thing to click instead of two files with the same name.
--
-- Rebuild after editing, from the repository root, and DELETE THE BUNDLE FIRST.
-- osacompile silently ignores -s when the .app already exists — it says "The -s option
-- will be ignored" and keeps the settings the old bundle had. An applet that is not
-- stay-open runs its quit handler the instant the run handler returns, and that handler
-- stops the server, so the workbench died about a second after the browser opened on
-- it. The flag it sets is OSAAppletStayOpen in the bundle's Info.plist; a test asserts
-- the shipped app carries it, because nothing else about the bundle looks wrong.
--   rm -rf "Junior Workbench.app"
--   osacompile -s -o "Junior Workbench.app" apps_and_interfaces/workbench_launcher.applescript

property logFile : "/tmp/junior_workbench.log"
property pidFile : "/tmp/junior_workbench.pid"

on run
	my startOrShow()
end run

-- Double-clicking an app that is already running never fires `on run` again; it fires
-- this. Without it, someone who closed the workbench tab and clicked the app to get it
-- back got nothing whatsoever.
on reopen
	my startOrShow()
end reopen

-- Every few seconds, once the run handler has returned. This is where a server that
-- died on its own gets noticed: an applet that outlives its server is not merely an
-- untidy process, it is an app that has stopped responding to being clicked.
on idle
	if not my isRunning() then quit
	return 5
end idle

on quit
	my stopServer()
	continue quit
end quit

on startOrShow()
	set repo to POSIX path of ((path to me as text) & "::")

	-- Already up: the click means "show me the workbench", not "start another one".
	if my isRunning() then
		set serverURL to my addressFromLog()
		if serverURL is not "" then my openInBrowser(serverURL)
		return
	end if

	set junior to my findJunior(repo)
	if junior is "" then
		activate
		display alert "Junior is not installed yet" message ("No Junior found in" & return & repo & return & return & "From a terminal, in that folder:" & return & "    python3 -m venv .venv" & return & "    .venv/bin/python -m pip install -e \".[app]\"") as critical
		quit
		return
	end if

	-- The whole launch goes inside a subshell whose own output is thrown away. Without
	-- that, `do shell script` never returns: it waits for the pipe it handed the shell
	-- to close, and the backgrounded server holds that pipe open for as long as it runs.
	try
		do shell script "cd " & quoted form of repo & " && rm -f " & logFile & " && ( nohup " & quoted form of junior & " workbench > " & logFile & " 2>&1 < /dev/null & echo $! > " & pidFile & " ) > /dev/null 2>&1"
	on error errText
		activate
		display alert "Junior could not start" message errText as critical
		quit
		return
	end try

	-- Wait for it to say where it is. Twenty seconds covers a cold torch import.
	set serverURL to ""
	repeat 40 times
		delay 0.5
		set serverURL to my addressFromLog()
		if serverURL is not "" then exit repeat
	end repeat

	if serverURL is "" then
		activate
		display alert "Junior started but never said where" message ("Log: " & logFile & return & return & my tailLog()) as critical
		my stopServer()
		quit
		return
	end if

	-- Junior prints its address a couple of seconds before it finishes binding the port,
	-- so trusting that line means the browser lands on "cannot connect". Ask the port
	-- itself whether it is answering yet, and only then open anything.
	if not my waitUntilAnswering(serverURL) then
		activate
		display alert "Junior never finished starting" message ("It said it would be at " & serverURL & " but never answered there." & return & return & "Log: " & logFile & return & return & my tailLog()) as critical
		my stopServer()
		quit
		return
	end if

	-- The CLI asks the browser to open too. This is the belt: Python's webbrowser can
	-- return without opening anything when it finds no handler, and says nothing.
	my openInBrowser(serverURL)
	display notification ("Running at " & serverURL & " — quit this app to stop it.") with title "Junior Workbench"
end startOrShow

on findJunior(repo)
	-- The checkout's own virtualenv, in the order a developer here would have made
	-- one. No $PATH candidate: see the note at the top.
	repeat with candidate in {".venv-arm64/bin/junior", ".venv/bin/junior"}
		set fullPath to repo & (candidate as text)
		try
			do shell script "test -x " & quoted form of fullPath
			return fullPath
		end try
	end repeat
	return ""
end findJunior


on waitUntilAnswering(theURL)
	-- A refused connection makes curl exit non-zero, which AppleScript raises as an
	-- error, so the try is the test. Thirty seconds covers a cold start.
	repeat 60 times
		try
			do shell script "curl -s -o /dev/null --max-time 2 " & quoted form of theURL
			return true
		end try
		delay 0.5
	end repeat
	return false
end waitUntilAnswering


on openInBrowser(theURL)
	try
		do shell script "open " & quoted form of theURL
	end try
end openInBrowser

on addressFromLog()
	try
		return do shell script "grep -o 'http://127.0.0.1:[0-9]*' " & logFile & " | tail -1"
	on error
		return ""
	end try
end addressFromLog

on tailLog()
	try
		return do shell script "tail -20 " & logFile
	on error
		return "(no log yet)"
	end try
end tailLog

on isRunning()
	try
		do shell script "kill -0 $(cat " & pidFile & ")"
		return true
	on error
		return false
	end try
end isRunning

on stopServer()
	try
		do shell script "PID=$(cat " & pidFile & "); pkill -P $PID 2>/dev/null; kill $PID 2>/dev/null; rm -f " & pidFile
	end try
end stopServer
