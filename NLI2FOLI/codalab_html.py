#!/usr/bin/env python
# -*- coding: utf8 -*-

''' Computational semantics course @ RUG-2019
    Lecturer: L.Abzianidze@rug.nl
    Assignment 4: Natural language inference with first-order logic theorem proving
'''

def codalab_html(scores, cm, filename):
    '''Write an html file with scores and a confusion matrix
    '''
    # internal css style and other styling
    (acc, prec, rec) = scores
    style = ("table, th, td { border: 1px solid black; border-collapse: collapse;}\n"
             "td {text-align: right;}\n"
             ".ltd {text-align: left;}\n"
             ".lab, th {background-color: #648ca8; color: #fff;}\n"
             "th, td {padding: 2px 10px 2px 10px;}\n"
             ".diag {background-color: #77dd77;}\n"
            )
    header = '<head>\n<title>Detailed results</title>\n<style>\n{}\n</style>\n</head>'.format(style)
    tag = 'p'
    # create a confusion table
    body = '<table>\n'
    body += '<tr>\n<th style="color:black">Gold \ Pred</th>\n<th>C</th>\n<th>E</th>\n<th>N</th>\n</tr>\n'
    for g in ['CONTRADICTION', 'ENTAILMENT', 'NEUTRAL']:
        body += '<tr>\n<td class="ltd lab"><b>{}<b></td>\n'.format(g)
        for p in 'CEN':
            sty = ' class="diag"' if g[0] == p else ''
            body += '<td{}>{}</td>\n'.format(sty, cm[(g[0],p)])
        body += '</tr>\n'
    body += '</table>\n'
    # generating the content
    body += '<{0}>Accuracy: <b>{1}</b></{0}>\n'.format(tag, round(acc, 4))
    body += '<{0}>Precision: <b>{1}</b></{0}>\n'.format(tag, round(prec, 3))
    body += '<{0}>Recall: <b>{1}</b></b></{0}>\n'.format(tag, round(rec, 3))
    html = '<!DOCTYPE html>\n<meta charset=utf-8>\n<html>\n{}\n<body>\n{}\n</body>\n</html>'.format(header, body)
    with open(filename, 'w') as f:
        f.write(html)
