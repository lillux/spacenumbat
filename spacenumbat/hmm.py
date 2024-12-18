#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:06:08 2024

@author: lillux
"""
import numpy as np
import pandas as pd
from spacenumbat.dist_prob import fit_lnpois, dnbinom, dpoilog
from spacenumbat.utils import fit_snp_rate, annot_segs
import tqdm


def viterbi_loh(HMM):
    n = len(HMM['x'])
    m = HMM['Pi'][:,:,1].shape[0]
    nu = np.zeros([n,m])
    #mu = np.zeros([n,m+1])
    z = np.zeros(n)
    
    nu[0,:] = np.log(HMM['delta'])
    logPi = np.log(HMM['Pi'])

    for i in range(n):
        if i > 0:
            matrixnu = np.full([m,m], nu[i-1,])
            nu[i,] = np.max(matrixnu + logPi[:,:,i], axis=1)
        nu[i,] = nu[i,] + dnbinom(x=HMM['x'][i], mu=HMM['pm']*HMM['pn'][i], size=HMM['snp_sig'])
        nu[i,] = nu[i,] + dpoilog(x = np.repeat(HMM['y'][i], m),
                                    sig = np.repeat(HMM['sig'], m),
                                    mu = HMM['mu'] + np.log(HMM['phi'] * HMM['d'] * HMM['lambda_star'][i]),
                                 log=True)

    z[-1] = np.argmax(nu[-1,])
    for i in range(n-2,-1,-1):
        z[i] = np.argmax(logPi[:,:,i+1][:,np.int32(z[i+1])] + nu[i,])
    z = z.astype(np.int32)
    #LL = np.max(nu[n-1,])
    HMM['states'] = np.array(HMM['states'])[z]

    return HMM['states']


def detect_clonal_loh(bulk, t:float=1e-5, snp_rate_loh:float=5, min_depth=0):

    bulk = bulk[(~bulk.loc[:, 'lambda_ref'].isna()) & (~bulk.loc[:,'gene'].isna())].copy()
    
    bulk_snps = {'CHROM':[],
             'gene':[],
             'gene_start':[],
             'gene_end':[],
             'gene_snps':[],
             'Y_obs':[],
             'lambda_ref':[],
             'logFC':[],
             'd_obs':[],
             'gene_length':[]}
    
    chrom_unique = bulk.CHROM.unique()
    for chrom in tqdm.tqdm(chrom_unique):
        gene_unique = bulk[bulk.loc[:, 'CHROM'] == chrom].gene.unique()
        for gene in gene_unique:
            tmp_bulk = bulk[(bulk.loc[:,'CHROM'] == chrom) &
                            (bulk.loc[:,'gene'] == gene)]
            gene_snps = tmp_bulk[(~tmp_bulk.loc[:,'AD'].isna()) &
                                (tmp_bulk.loc[:,'DP'] > min_depth)].shape[0]
            Y_obs = tmp_bulk.loc[:,'Y_obs'].dropna().unique().astype(np.int32).sum()
            lambda_ref = tmp_bulk.loc[:,'lambda_ref'].dropna().unique().item()
            logFC = tmp_bulk.loc[:,'logFC'].dropna().unique().item()
            d_obs = tmp_bulk.loc[:,'d_obs'].dropna().unique().item()
            bulk_snps['gene_snps'].append(gene_snps)
            bulk_snps['Y_obs'].append(Y_obs)
            bulk_snps['lambda_ref'].append(lambda_ref)
            bulk_snps['logFC'].append(logFC)
            bulk_snps['d_obs'].append(np.int32(d_obs))
            bulk_snps['gene'].append(tmp_bulk.gene.values[0])
            bulk_snps['gene_start'].append(tmp_bulk.gene_start.astype(np.int32).values[0])
            bulk_snps['gene_end'].append(tmp_bulk.gene_end.astype(np.int32).values[0])
            bulk_snps['gene_length'].append(np.array(tmp_bulk.gene_end.values[0] - tmp_bulk.gene_start.values[0]).astype(np.int32))
            bulk_snps['CHROM'].append(chrom)
    
    bulk_snps_df = pd.DataFrame(bulk_snps)
    bulk_snps_df = bulk_snps_df[(bulk_snps_df.logFC < 8) & (bulk_snps_df.logFC > -8)]
    bulk_snps_df = bulk_snps_df.sort_values(['CHROM','gene_start']).reset_index(drop=True)
    
    fit = fit_lnpois(bulk_snps_df.Y_obs.values,
                     bulk_snps_df.lambda_ref.values,
                     bulk_snps_df.d_obs.unique())
    
    mu, sig = fit
    bulk_snps_df.gene_length = bulk_snps_df.gene_length.values.astype(np.int32)
    snp_fit = fit_snp_rate(bulk_snps_df.gene_snps.values, bulk_snps_df.gene_length.values)
    snp_rate_ref, snp_sig = snp_fit
    
    n = bulk_snps_df.shape[0]
    A = np.array([[1-t, t],[t, 1-t]])
    As = np.tile(A, (n, 1, 1)).transpose(1, 2, 0)
    
    HMM = {
    'x': bulk_snps_df.gene_snps,
    'Pi': As,
    'delta': [1-t, t],
    'pm': np.array([snp_rate_ref, snp_rate_loh]),
    'pn': np.array(bulk_snps_df.gene_length / 1e6),
    'snp_sig': snp_sig,
    'y': bulk_snps_df.Y_obs,
    'phi': [1, 0.5],
    'lambda_star': bulk_snps_df.lambda_ref,
    'd': bulk_snps_df.d_obs.unique(),
    'mu': mu,
    'sig':sig,
    'states': ['neu', 'loh']
    }
    
    vtb = viterbi_loh(HMM)
    bulk_snps_df.loc[:,'cnv_state'] = vtb
    
    bulk_snps_df.loc[:,'snp_index'] = bulk_snps_df.index
    bulk_snps_df.loc[:,'POS'] = bulk_snps_df.gene_start
    bulk_snps_df.loc[:,'pAD'] = 1

    segs_loh = annot_segs(bulk_snps_df)  # you may want this output for plot in bulk

    snp_rate = []
    for chrom in segs_loh.CHROM.unique():
        chrom_bulk = segs_loh[segs_loh.CHROM == chrom]
        for seg in chrom_bulk.seg.unique():
            seg_bulk = chrom_bulk[chrom_bulk.seg == seg]
    
            snp_rate.append(fit_snp_rate(seg_bulk.gene_snps, seg_bulk.gene_length)[0])
    
    segs_loh = segs_loh.groupby(['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state'], observed=True, sort=False).sum()
    segs_loh = segs_loh.reset_index().loc[:,['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state', 'gene_snps', 'gene_length']]
    
    segs_loh.loc[:,'snp_rate'] = snp_rate
    segs_loh = segs_loh[segs_loh.cnv_state == 'loh'].copy()
    segs_loh.loc[:, 'loh'] = True
    segs_loh = segs_loh.reset_index().loc[:,['CHROM', 'seg', 'seg_start', 'seg_end', 'snp_rate', 'loh']]
    
    if segs_loh.shape[0] == 0:
        segs_loh = None
    return segs_loh

    return segs_loh