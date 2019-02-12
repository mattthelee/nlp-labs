# -*- coding: utf-8 -*-
# Python version of SemEval 2017 Perl-based scorer: Matthew Purver 2019
# Original information:
#
#  Author: Preslav Nakov
#  
#  Description: Scores SemEval-2017 task 4, subtask A
#               Calculates macro-average R, macro-average F1, and Accuracy
#
#  Last modified: February 11, 2017
#
#

from collections import defaultdict, Counter
import unicodecsv
import sys

CLASSES = ['positive', 'negative', 'neutral']

def main(argv):

    (goldFile, inputFile) = argv[1:3]
    outputFile = inputFile+'.scored'

    stats = defaultdict(Counter)

    ### 1. Read the files and get the statsitics
    with open( inputFile, 'rb') as inf, open( goldFile, 'rb' ) as goldf:
        inreader = unicodecsv.reader(inf, encoding='utf-8', delimiter='\t')
        goldreader = unicodecsv.reader(goldf, encoding='utf-8', delimiter='\t')

        for inline in inreader:
            try:
	        ### 1.1. Check the input file format
	        #1	positive	i'm done writing code for the week! Looks like we've developed a good a** game for the show Revenge on ABC Sunday, Premeres 09/30/12 9pm
                proposedId = int(inline[0])
                proposedLabel = inline[1]
                assert proposedLabel in CLASSES
            except Exception:
                raise FormatError("Wrong file format for %s: %s" (inputFile, inline))
            
            goldline = goldreader.next()
            try:
	        ### 1.2	. Check the gold file format
	        #NA	T14114531	positive
                goldId = goldline[0]
                trueLabel = goldline[1]
                assert trueLabel in CLASSES
            except Exception:
                raise FormatError("Wrong file format for %s: %s" (goldFile, goldline))

            ### 1.3. Update the statistics
            stats[proposedLabel][trueLabel] += 1


    ### 2. Initialize zero counts
    # (not required - done by Counter)

    ### 3. Calculate the F1
    with open( outputFile, 'wb' ) as outf:
        writer = unicodecsv.writer(outf, encoding='utf-8')

        writer.writerow([inputFile])

        avgR  = 0.0
        avgF1 = 0.0
        diag = 0.0
        for c in CLASSES:
        
	    denomP = stats[c]['positive'] + stats[c]['negative'] + stats[c]['neutral']
            denomP = (denomP if (denomP > 0) else 1)
	    P = float(stats[c][c]) / denomP

	    denomR = stats['positive'][c] + stats['negative'][c] + stats['neutral'][c]
            denomR = (denomR if (denomR > 0) else 1)
	    R = float(stats[c][c]) / denomR
			
	    denom = ((P+R) if (P+R > 0) else 1)
	    F1 = 2*P*R / denom

            avgR += R
	    if (c != 'neutral'): avgF1 += F1
	    writer.writerow([c, 'P', P, 'R', R, 'F1', F1])
	    print([c, 'P', P, 'R', R, 'F1', F1])

            diag += stats[c][c]

        avgR /= float(len(CLASSES))
        avgF1 /= float(len(CLASSES)-1)
        acc = diag / sum([sum(s.values()) for s in stats.values()])

        writer.writerow(['AvgR_3',avgR,'AvgF1_2',avgF1,'Acc',acc])
        writer.writerow(['OVERALL SCORE',avgR])
        print(['AvgR_3',avgR,'AvgF1_2',avgF1,'Acc',acc])
        print(['OVERALL SCORE',avgR])


if __name__ == "__main__":
    main(sys.argv)
