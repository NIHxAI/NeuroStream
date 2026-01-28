import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from matplotlib import ticker

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.io
import plotly.graph_objects as go
import plotly.express

from scipy.stats import (
    t,
    ttest_ind,
    f_oneway
)
from pingouin import ptests
from sklearn.decomposition import IncrementalPCA as PCA

palette=sns.color_palette('Set1')

code={
    'selected':
        {'left':'Group','center':'X','right':'Y'},
    'gender':
        {1:'Male',2:'Female'},
    'age_group':
        {0:'40s',1:'50s',2:'60s',3:'70s',4:'80s'},
    'edu_level':
        {0:'None',1:'Elementary',2:'Middle',3:'High',4:'College',5:'University',6:'Above University',99999:'Unknown'},
    'htn':
        {1:'HTN',2:'Non-HTN'},
    'dm':
        {1:'DM',2:'Non-DM'}
}

plot_style_constraint=dict(
    zeroline=False,
    zerolinewidth=1,
    zerolinecolor='lightgray',
    showline=True,
    linecolor='lightgray',
    gridcolor='lightgray'
)

plt.rcParams['font.family']='Serif'


def get_datetime_str()->str:
    return 


def claim(title:str)->None:
    dt=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt, ':' ,title)
    return 


def lap(func,**kwargs):
    t0=time()
    result=func(**kwargs)
    claim(f'{func.__repr__()} took {time()-t0:.1f} s')
    return result


def sanitise(q:str)->str:
    if (q!='sex' and len(q)<3):
        return q.upper()

    elif (q=='mmse'):
        return q.upper()

    elif (q=='htn'):
        return 'HTN'

    elif (q[:4]=='APOE'):
        return 'APOE OPI'

    elif (q[:3]=='icv'):
        return 'ICV'
    
    if q[-1].isnumeric():
        q=q.split('_')
        w=' '.join([w for w in q if all(map(lambda q: q.isalpha(), w))]).title()
        e=''.join([r for r in q if all(map(lambda q: q.isnumeric(), r))])
        return '{} ({})'.format(w, e)
    else:
        return ' '.join(q.split('_')).title()

    return 'column-name-parsing-failure'


def tagClassObsCount(
    noe:pd.DataFrame,
    cat:str
)->dict:

    obs_castable=np.can_cast(noe.loc[:,cat].values,'f8')
    obs_count=noe.loc[:,cat].value_counts().to_dict()
    
    if obs_castable:
        obs_count_tag_view={q:f'{code[cat][q]}<br>n={w}' for q,w in obs_count.items()}
    else:
        obs_count_tag_view=noe.loc[:,cat].value_counts().to_dict()
        for obs_view in obs_count_tag_view:
            obs_count_tag_view[obs_view]=f'{obs_view}<br>n={obs_count_tag_view[obs_view]}'
    
    claim(f'{obs_count_tag_view=}')
    
    return obs_count_tag_view


def getNoeImage(
    noeImage:dict,
    selectionState
):

    _orgVarName=selectionState[0]
    if _orgVarName in noeImage.keys():
        return noeImage[_orgVarName]
    else:
        return None


def projectNoeImage(graph,image)->go.Figure:

    graph.add_layout_image(dict(
        source=image,
        xref='paper',
        yref='paper',
        x=0.85,
        y=0.25,
        sizex=.10,
        sizey=.10,
        opacity=.7
    ))

    return graph


def multiBox(
    noe:pd.DataFrame,
    c:tuple,
    y:tuple,
    vs:bool
)->tuple[str, go.Figure]:
    noe=noe.sort_values(c[0],ascending=False)
    _title=f'{y[1]}<sup>{y[2]}</sup> by {c[1]}'

    def _titling():
        tStat=_tt(noe,c[0],y[0])
        tStatStr=f't={tStat[0]:.2f}, p={tStat[1]:.2f}'
        
        es=_getEs(noe,c[0],y[0])
        esSize=noe.groupby(c[0])[y[0]].count()
        esCi=_getEsci(
            noe,
            na=esSize.iat[0],
            nb=esSize.iat[1],
            d=es,
            ci=.99
        )
        esStr=f'es={es:.2f} [{esCi[0]:.2f}, {esCi[1]:.2f}]'
        
        return (_title,tStatStr,esStr)
    
    cats=set(q for q in noe.loc[:,c[0]].values)
    noe=noe.loc[:,[c[0],y[0]]]
    title=_titling() if vs else _title
    categoricalObsCount=tagClassObsCount(noe, c[0])
    
    graph=go.Figure()
    
    for cat in cats:
        graph.add_trace(
            go.Box(
                y=noe.loc[noe[c[0]]==cat,y[0]],
                name=f'{categoricalObsCount[cat]}',
            )
        )
    
    graph.update_layout(
        legend_orientation='v',
        legend_title=f'{c[1]}',
        xaxis={
            'title':f'{c[1]}',
            **plot_style_constraint
        },
        yaxis={
            'title':f'{y[1]}',
            **plot_style_constraint
        },
        margin={
            't':0,
            'b':0
        }
    )
    
    return title,graph


def _f(
    noe:pd.DataFrame,
    c:tuple,
    y:tuple
)->tuple:

    x=noe.loc[:,[c[0],y[0]]]
    cats=x.loc[:,c[0]].unique()
    
    fTestResult=tuple(q for q in f_oneway(
        *[x.loc[x[c[0]]==q,y[0]] for q in cats],
        nan_policy='omit'
    ))

    return fTestResult


def sign(x):
    if len(x)>0 and x[0]=='*':
        return 'background-color:#d9d9d9'
    if not x:
        return 


def intergroupTt(
    noe:pd.DataFrame,
    c:tuple,
    y:tuple,
    star=True
)->tuple[str,pd.DataFrame]:

    x=pd.pivot(
        noe.loc[:,[c[0],y[0]]],
        columns=c[0],
        values=y[0]
    )
    x.columns.name='Group'
    
    intergroupTtestResult=ptests(x,stars=star).map(lambda q:'NS' if not q else q)
    intergroupTtestResult.columns=[code[c[0]][q] for q in intergroupTtestResult.columns]
    intergroupTtestResult.index=[code[c[0]][q] for q in intergroupTtestResult.index]
    intergroupTtestResultTitle=f'##### Groupwise T-test: {y[1]}<sup>{y[2]}</sup> by {c[1]}<br><sub>* p<0.05, ** <0.01, *** <0.001, or Not Significant</sub>'
    
    return intergroupTtestResultTitle,intergroupTtestResult


def scatterTrajectory(
    noe,
    c,
    x,
    y,
    traject='lowess',
)->tuple[str,go.Figure]:

    if x[0]==y[0]:
        raise Exception(f"You're attempting to assign same variable '{x[0]}' to both axes.")
    
    # Todo for to cope with go type inference
    _noe=pd.concat([
        noe.loc[:,c[0]].astype('object'),
        noe.loc[:,[y[0],x[0]]]
    ],axis=1).sort_values(c[0],ascending=False)
    
    title=f'##### Scatterplot: {x[1]}<sup>{x[2]}</sup> and {y[1]}<sup>{y[2]}</sup> by {c[1]}'
    
    graph=plotly.express.scatter(
        _noe,
        color=c[0],
        y=y[0],
        x=x[0],
        trendline=traject,
        opacity=.5
    )
    
    categoricalObsCount=tagClassObsCount(noe, c[0])
    
    # Todo for to cope with go type inference
    graph.for_each_trace(
        lambda q:q.update(name=categoricalObsCount[int(q.name) if q.name.isnumeric() else q.name])
    )

    graph.update_layout(
        legend_orientation='v',
        legend_title=f'{c[1]}',
        xaxis={
            'title':f'{x[1]}',
            **plot_style_constraint
        },
        yaxis={
            'title':f'{y[1]}',
            **plot_style_constraint
        },
        margin={
            't':0,
            'b':0
        }
    )
    
    return title,graph


def decompose(
    noe:pd.DataFrame,
    c:str,
    y:list,
    dim:int=2
)->tuple:
    
    noe=noe.dropna()
    
    cats=noe.loc[:,c[0]] # Indexer
    cat=np.sort(cats.unique()) # Categorical
    
    tag=list(
        zip(cat,range(len(cat)),palette[:cat.size])
    )
    
    Reducer=PCA(batch_size=1024)
    
    i=noe.loc[:,y]
    
    x=Reducer.fit_transform(i)

    explainedRatio=(Reducer.explained_variance_ratio_)

    categorical_count=tagClassObsCount(noe, c[0])

    fig,ax=plt.subplots(figsize=(4,4))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    labels=[]
    for t in tag:
        ax.scatter(
            x[cats==t[0],0],
            x[cats==t[0],1],
            label=t[0],
            color=t[2],
            s=10,
            alpha=.7,
            edgecolor='none'
        )
        labels.append(categorical_count[t[0]].replace('<br>','\n'))
    
    ax.set_xlabel(f'PC1 ({explainedRatio[0]*100:.1f}%)',fontsize=6)
    ax.set_ylabel(f'PC2 ({explainedRatio[1]*100:.1f}%)',fontsize=6)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    
    ax.legend(
        title=f'{c[1]}',
        labels=labels,
        loc='upper right',
        facecolor='#F0F0F0',
        framealpha=.5,
        fontsize=5,
        title_fontsize=5,
    )
    
    return fig,Reducer


def get_length(thinner):
    ratio=32/193
    return ratio * len(thinner)


def draw_violin(
    deta,
    value_column,
    group_column='cohort'
):

    categoricalObsCount=tagClassObsCount(deta, group_column)

    data=deta.melt(group_column,value_column,'Variable','Value')
    
    cats=data[group_column]
    
    fig,ax=plt.subplots(figsize=(3,get_length(value_column)))
    
    sns.violinplot(
        formatter=lambda q:q.replace('_','-')[:q.rfind('_')].title(),
        data=data,x='Value',y='Variable',hue=group_column,width=1,inner=None,split=True,cut=1,
        linewidth=.1,palette=palette[:cats.nunique()],orient='h'
    )
    
    ax.axvline(x=0,alpha=.5,color='#303030',linewidth=.5,linestyle='--')
    
    ax.yaxis.set_tick_params(labelsize=4,labelrotation=20)
    ax.xaxis.set_tick_params(labelsize=7)
    
    ax.legend(
        title=group_column.title(),
        labels=[q.replace('<br>','\n') for q in categoricalObsCount.values()],
        loc='upper right',
        facecolor='#F0F0F0',
        framealpha=.5,
        fontsize=5,
        title_fontsize=5,
    )
    
    
    sns.despine(top=True,right=True)

    
    return fig,ax


def _tt(
    noe:pd.DataFrame,
    cat:str,
    var:str,
)->tuple:

    cats=noe.loc[:,cat].unique()

    sieves={q:noe.loc[:,cat].eq(q) for q in cats}
    sieved=tuple(noe.loc[sieves[cat],var] for cat in cats)
    
    return tuple(q for q in ttest_ind(*sieved,nan_policy='omit'))


def _getEs(
    noe,
    cat:str,
    var:str
):

    x=noe.groupby(cat)[var].agg(['mean','std']).T
    muDiff=np.diff(x.loc['mean',:].to_numpy(np.float32))
    sigmaSqrt=np.sqrt(
        np.sum(x.loc['std',:].to_numpy(np.float32)**2)/2
    )
    
    return (muDiff/sigmaSqrt)[0]


def _getEsci(
    noe,
    na,
    nb,
    d,
    ci=.99
):
    size=na+nb
    sizes=na*nb
    
    se=np.sqrt(
        (size/sizes) + d**2/(size*2)
    )
    dof=size-2
    critical=t.ppf(
        q=(1-ci)/2,
        df=dof
    )
    
    return np.sort(
        np.array([d-(critical*se),d+(critical*se)],np.float32)
    )


def isVs(x:pd.Series)->bool:
    return len(x.sample(frac=.1).unique())==2
