#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 13:55:40 2018

@author: tghaddar
"""

def get_ijk(ss_id,num_row,num_col,num_plane):
#  k = ss_id%num_plane
#  j = int((ss_id - k)/num_plane%num_row)
#  i = int(((ss_id-k)/num_plane - j)/num_row)
  k = int(ss_id/(num_row*num_col))
  if (ss_id >= num_row*num_col):
    ss_id -= (k)*num_row*num_col
  j = int(ss_id % num_row)
  i = int((ss_id - j)/num_row)
  
  
  return i,j,k

def get_ij(ss_id,numrow,numcol):
  j = int(ss_id % numrow)
  i = int((ss_id - j)/numrow)
  return i,j

def get_ss_id(i,j,numrow):
  return i*numrow + j
  