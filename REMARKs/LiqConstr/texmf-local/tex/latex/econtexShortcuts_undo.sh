#!/bin/bash
# Script to replace the various shortcuts defined in econtexShortcuts.sty with the destination
# This is necessary to convert to latex code that will run on systems without the econtex system installed

if [ $# -ne 2 ]; then
    echo "Usage: `basename $0` /Volumes/Data/Courses/Choice/LectureNotes/Consumption/Handouts/PerfForesightCRRA/LaTeX/PerfForesightCRRA.tex /Volumes/Data/Courses/Choice/LectureNotes/Consumption/Handouts/PerfForesightCRRA/LaTeX" 
    echo "$0 /Volumes/Data/Courses/Choice/LectureNotes/Consumption/Handouts/PerfForesightCRRA/LaTeX/PerfForesightCRRA.tex /Volumes/Data/Courses/Choice/LectureNotes/Consumption/Handouts/PerfForesightCRRA/LaTeX" | pbcopy
    exit 1
fi

file=$1
dest=$2

fileBase=$(basename -- "$file")
fileNoExt="${fileBase%.*}"
filePath="$(dirname "$file")"

# echo $file
# echo $fileBase
# echo $fileNoExt
# echo $filePath


# echo Processing $filePath/$fileName

grep providecommand econtexShortcuts.sty > /tmp/providecommands.txt

#echo "\providecommand{\tNow}{\ensuremath{t}}" > /tmp/providecommands.txt

#echo "\providecommand{\WNet}{\ensuremath{X}} % Total wealth" > /tmp/providecommands.txt
#echo "\providecommand{\VE}{{V}^{e}}" > /tmp/providecommands.txt
#echo "\providecommand{\vFirm}{\ensuremath{\mathrm{e}}}" > /tmp/providecommands.txt

cat /tmp/providecommands.txt  | awk -F '%' '{print $1}' > /tmp/nocomments.txt

# echo nocomments

cat /tmp/nocomments.txt | sed 's/[[:space:]]*$//' > /tmp/removeeolwhitespace.txt

# cat /tmp/removeeolwhitespace.txt | sed 's/\\ensuremath//' > /tmp/replacedensuremath.txt
cat /tmp/removeeolwhitespace.txt | sed 's/\\ensuremath{\(.*\)}/\1/'  > /tmp/replacedensuremath.txt

#echo replacedensuremath

cat /tmp/replacedensuremath.txt | sed 's/{/,/' > /tmp/beginbracketstocommas.txt

#echo beginbracketstocommas

cat /tmp/beginbracketstocommas.txt | sed 's/}{/,/' > /tmp/connectingbracketstocommas.txt

#echo connectingbracketstocommas
cat /tmp/connectingbracketstocommas.txt | sed 's/.$//' > /tmp/deleteendbrackets.txt

#echo deleteendbrackets

cat /tmp/deleteendbrackets.txt | grep    ',{' > /tmp/commandstillhasbraces.txt

# echo commandstillhasbraces
# cat /tmp/deleteendbrackets.txt | grep -v ',{' > /tmp/commandswithnobraces.txt

# echo commandswithnobraces
# cat /tmp/commandstillhasbraces.txt | sed 's/{\(.*\)$/\1/' > /tmp/removefirstbrace.txt

# echo removefirstbrace
# cat /tmp/removefirstbrace.txt | sed 's/.$//' > /tmp/removelastbrace.txt

# echo removelastbrace

cd $filePath

rm -f $dest/$fileNoExt-Plaim-Make.sh
echo "#/bin/bash" > $dest/$fileNoExt-Plain-Make.sh

rm -f $dest/econtexShortcuts_used.sty 

# #awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn  }' /tmp/deleteendbrackets.txt
# #awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn  }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
# #awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" " }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
# #awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" " }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
# awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" | awk \"{ print $4 }\"" }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
# awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" | awk '\''%s%'\''{ print $4 }\"" }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
#awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" | awk '\''{ print $4 }'\''" }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
#awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" | awk '\''{ print $4 }'\'' > /tmp/matches.txt ; matches=`cat /tmp/matches.txt` ; if [ \"$matches\" -ne \"0\" ]; then echo nonzero ; fi "}' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
awk -v fileIn="$fileNoExt.tex" -v fileOut="$dest/econtexShortcuts_used.sty" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\" | awk '\''{ print $4 }'\'' > /tmp/matches.txt ; matches=`cat /tmp/matches.txt` ; if [ \"$matches\" -ne \"0\" ]; then echo \""$1 "\{" $2 "\}"   "\{" $3  "\}" " \" >> " fileOut " ; fi "}' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh
# awk -v fileIn="$fileNoExt.tex" -F',' '{ if ($3) print "rpl -vs \""$2"\" \""$3"\" " fileIn " &> /tmp/results.txt ; cat /tmp/results.txt | grep \"A Total of\"  " }' /tmp/deleteendbrackets.txt >> $fileNoExt-Plain-Make.sh



chmod a+x $dest/$fileNoExt-Plain-Make.sh
# cat ./$file-Plain.sh

echo 'A bash script that will convert '$fileNoExt.tex' to '$fileNoExt-Plain.tex ' should be at '$dest' with the name '$fileNoExt-Plain-Make.sh
echo ''
head $dest/$fileNoExt-Plain-Make.sh

# echo Hit return to run the command
# cmd="$dest/$fileNoExt-Plain-Make.sh"
# echo $cmd
# read answer
#eval $cmd
#cat $dest/$fileNoExt-Plain-Make.sh
#cat $dest/$fileNoExt-Plain-Make.sh | pbcopy


