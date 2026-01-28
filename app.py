import os
import pandas as pd
import streamlit as st
import util
import deta

from types import SimpleNamespace

repo_path="./assets"
dog="dog"
grouper="cohort"
title="Analysis of Quantitized Data of Brain MR Images"

st.set_page_config(
    page_title=title,
    page_icon=None
)

st.markdown(
    '''
        <style>
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {font-size:1.4rem;}
        </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    f"### {title}"
)

transform_method_constraint = {
    "None": "no transformation",
    "Log Transform (log1p)": "Log Transform (log1p)",
    "Log Transform + Z-score": "Log Transform + Z-score",
    "Scale (Z-score)": "Scale (Z-score)",
    "Divide by Intracranial Volume": "divided by intracranial volume",
    "Combat (covariate: gender, age)": "Combat (covariate: gender, age)",
}


noe,noeSite,noeCatName,noeVarName,noeVolName=deta.get_var()
noeImage=deta.get_noe_image(repo_path)

div_control_surface=st.container(border=True)
with div_control_surface:
    
    div_cohort_select=st.container()
    with div_cohort_select:
        msCenter=st.columns(1)[0]
        with msCenter:
            selectedUpper=st.multiselect(
                f"Cohort ({len(noeSite)})",
                noeSite,
                default=noeSite,
                max_selections=len(noeSite),
                help="Only variables existing on selected cohort(s) are shown, or included"
            )
    
    noe,noeSite,noeCatName,noeVarName,noeVolName=deta.get_var(
        noe.loc[noe.loc[:, grouper].map(lambda q:q in selectedUpper),:].dropna(axis=1)
    )
    
    div_control_options=st.container()
    with div_control_options:
        
        div_transform_method=st.columns(1)[0]
        with div_transform_method:
            if len(selectedUpper) > 1:
                transform_methods=transform_method_constraint.keys()
            else:
                transform_methods=[q for q in transform_method_constraint.keys() if q != 'Batch Correction']
            transform_method_selected=st.radio(
                "Preprocess Method",
                transform_methods,
                horizontal=True
            )
        
        div_select_switch,div_trim_switch=st.columns(2)
        with div_select_switch:
            select_switch=st.toggle(
                "Show Prominent Regions Only",
                value=False,
                help="List regions frequently mentioned in reseraches (White matter, Gray matter, Ventricles, Frontal lobe, etc)."
            )
        with div_trim_switch:
            trim_switch=st.toggle(
                "Trim Outliers by [.001, .999]",
                value=False
            )
        
        transform_method=transform_method_constraint[transform_method_selected]
        noe=deta.transform(noe,transform_method,noeVarName,noeVolName)
        noe,noeSite,noeCatName,noeVarName,noeVolName=deta.get_var(noe)
        
        if select_switch:
            noeVarName=[q for q in noeImage.keys() if q!="placeholder"]
            noeVolName=noeVarName
        if trim_switch:
            noe=deta.trim(noe,noeVolName)
        
    div_feature_select_surface=st.container()
    with div_feature_select_surface:
        
        dropdown_listbox_left,dropdown_listbox_center,dropdown_listbox_right=st.columns(3)
        with dropdown_listbox_left:
            noeCatName=noeCatName[:-1] if len(selectedUpper)==1 else noeCatName
            lt=st.selectbox(
                f"Group ({len(noeCatName)})",
                noeCatName,
                format_func=util.sanitise,
                key="l"
            )
        with dropdown_listbox_center:
            noeVarName=sorted(noeVarName)
            ct=st.selectbox(
                f"X ({len(noeVarName)})",
                noeVarName,
                format_func=util.sanitise,
                key="c"
            )
        with dropdown_listbox_right:
            rt=st.selectbox(
                f"Y ({len(noeVarName)})",
                noeVarName,
                index=3,
                format_func=util.sanitise,
                key="r"
            )
        
    Selected=SimpleNamespace()

    for select in zip(
        ("left","center","right"),
        (lt,ct,rt)
    ):
        setattr(
            Selected,
            select[0],
            (
                select[1],
                f"{util.sanitise(select[1])}",
                util.code['selected'][select[0]]
            )
        )

page_description,page_volumetry,page_upload=st.tabs([
    "EDA",
    "PCA",
    "Upload Dataset"
])

with page_description:
    canvas=st.container(border=True)
    with canvas:
        vs=util.isVs(noe.loc[:,Selected.left[0]])
        for selected in (Selected.center,Selected.right):
            st.markdown(f"##### {selected[1]}<sup>{selected[2]}</sup>",
                unsafe_allow_html=True)
            q,w,e,r=st.columns(4)
            for a in zip(
                (q,w,e,r),
                ("mean","median","std","count")
            ):
                if a[1]!="count":
                    a[0].metric(
                        a[1].title(),
                        f"{noe[selected[0]].agg(a[1]):.2f}"
                    )
                else:
                    a[0].metric(
                        a[1].title(),
                        f"{noe[selected[0]].agg(a[1]):0}"
                    )
            
            noeImageEach=util.getNoeImage(noeImage,selected)
            
            boxplotLayoutProportion=[7.8, 3.2, .1] if select_switch else 1
            
            boxplotImageDivider=st.columns(boxplotLayoutProportion)
            with boxplotImageDivider[0]:
                boxplotLeft=util.multiBox(
                    noe,
                    Selected.left,
                    selected,
                    vs=vs
                )
                
                boxplotTitle=boxplotLeft[0]
                if vs:
                    st.markdown(
                        f"##### Boxplots: {boxplotTitle[0]}<br><sub>{boxplotTitle[1]}, {boxplotTitle[2]}</sub>",
                        unsafe_allow_html=True
                    )
                
                else:
                    st.markdown(
                        f"##### Boxplots: {boxplotTitle}",
                        unsafe_allow_html=True
                    ) 
                
                st.plotly_chart(boxplotLeft[1])
            
            if select_switch:
                with boxplotImageDivider[1]:
                    st.markdown(
                        "<br><br><br><br><br>",
                        unsafe_allow_html=True
                    )
                    st.image(noeImageEach)
            
            if vs==False:
                div_groupwise_table=st.container()
                with div_groupwise_table:
                    intergroupTtestResult=util.intergroupTt(
                        noe,
                        Selected.left,
                        selected,
                    )
                    
                    st.markdown(
                        intergroupTtestResult[0],
                        unsafe_allow_html=True
                    )
                    
                    st.dataframe(
                        intergroupTtestResult[1].style.map(util.sign),
                        on_select="ignore",
                        use_container_width=True
                    )
            
            st.divider()
        
        graph_center=st.columns(1)[0]
        with graph_center:
            try:
                scatter=util.scatterTrajectory(
                    noe=noe,
                    c=Selected.left,
                    x=Selected.center,
                    y=Selected.right,
                )
            except Exception as err:
                st.exception(err)
            else:
                st.markdown(
                    scatter[0],
                    unsafe_allow_html=True
                )
                st.plotly_chart(scatter[1])

with page_volumetry:
    div_volumetry_plot=st.container(border=True)
    with div_volumetry_plot:
        decomposition_vol_name=[q for q in noeVolName if not q.startswith("icv")]
        decomposition_plot_title=f"##### Principal Component Analysis<br><sub>{len(decomposition_vol_name)} volume parameters, {transform_method}</sub>"
        
        st.markdown(
            decomposition_plot_title,
            unsafe_allow_html=True
        )
        decomposed=util.lap(
            util.decompose,
            noe=noe,
            c=Selected.left,
            y=decomposition_vol_name
        )
        st.pyplot(
            decomposed[0],
            use_container_width=True,
            transparent=True
        )
        
        st.divider()
        
        violin_plot_title=f"##### Batch Effect<br><sub>{len(decomposition_vol_name)} volume parameters, {transform_method}</sub>"
        st.markdown(
            violin_plot_title,
            unsafe_allow_html=True
        )
        st.pyplot(
            util.lap(
                util.draw_violin,
                deta=deta.transform(noe,"scale",noeVarName,decomposition_vol_name),
                value_column=decomposition_vol_name
            )[0],
                use_container_width=True,
            transparent=True
        )
        
        decomposer=decomposed[1]


if 'upload_done' not in st.session_state:
    st.session_state['upload_done'] = False
    
with page_upload:
    div_uploader = st.container(border=True)
    with div_uploader:
        st.markdown('##### Upload Dataset')
        uploaded_file = st.file_uploader(
            label='Feed Dataset',
            label_visibility='hidden',
            type='csv',
            key='uploader_',
        )
        if uploaded_file and not st.session_state['upload_done']:
            try:
                pd.read_csv(
                    uploaded_file,
                    na_filter=False
                ).to_feather(
                    os.path.join(deta.path, uploaded_file.name.replace('.csv', '.f')),
                    compression='zstd',
                    compression_level=9,
                )
                st.session_state['upload_done'] = True
                st.success(f'Upload Success ({uploaded_file.name})')
                st.rerun()
            except Exception as err:
                st.error(f'Unsupported File: {err.__str__()}')


foot=st.container(border=False)
with foot:
    st.markdown(
        """**© 2026 <https://www.nih.go.kr>**""",
        unsafe_allow_html=True
    )
