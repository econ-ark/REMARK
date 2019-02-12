#!/bin/sh

if [ $# -eq 0 ]
then
  echo "usage: ${0##*/} <handoutName>"
  exit 1
fi

echo "\\Preamble{}"       > $1.cfg
printf '\\begin{document}' >> $1.cfg
# printf "\\\provideboolean{tex4htCDC}\\\setboolean{tex4htCDC}{true}"       > $1.cfg
# printf '\\begin{document}' >> $1.cfg
echo "\HCode{         " >> $1.cfg
printf "<meta name = \042Author\042      content = \042Christopher D. Carroll\042> \Hnewline \n" >>$1.cfg
printf "<meta name = \042Description\042 content = \042" >>$1.cfg
[[ -e $1.title ]] && (cat $1.title | tr -d '\012') >> $1.cfg
[[ -e $1.title ]] && printf "\042> \Hnewline \n" >>$1.cfg
[[ -e $1.title ]] && printf "<title>"  >> $1.cfg 
[[ -e $1.title ]] && (cat $1.title | tr -d '\012') >> $1.cfg
[[ -e $1.title ]] && printf "</title> \Hnewline" >> $1.cfg
echo "}" >> $1.cfg
printf '\\EndPreamble' >> $1.cfg
echo '' >> $1.cfg


