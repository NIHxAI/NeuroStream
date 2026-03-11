import os
import pandas as pd
import streamlit as st
import util
import deta

from types import SimpleNamespace

repo_path="./assets"
grouper="cohort"
title="NeuroStream"

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
    "Combat": "Combat", 
}

try:
    noe, noeSite, noeCatName, noeVarName, noeVolName = deta.get_var()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.warning("⚠️ No dataset found. Please upload a .f file in the 'Upload' tab.")
if not data_loaded:
    st.info("Please upload a dataset to begin.")
    

if data_loaded:
    noe, noeSite, noeCatName, noeVarName, noeVolName = deta.get_var()
    noeImage=deta.get_noe_image(repo_path)
    noe_before_preprocess = noe.copy()
    
    div_control_surface=st.container(border=True)
    with div_control_surface:
        
        div_cohort_select=st.container()
        with div_cohort_select:
            msCenter=st.columns(1)[0]
            with msCenter:
                selectedUpper = st.multiselect(
                    f"Selected Cohorts",
                    noeSite,
                    default=noeSite,
                    key="cohort_selection"
                )
    
            st.caption(f"📍 Currently comparing **{len(selectedUpper)}** out of {len(noeSite)} cohorts.")
        noe,noeSite,noeCatName,noeVarName,noeVolName=deta.get_var(
            noe.loc[noe.loc[:, grouper].map(lambda q:q in selectedUpper),:].dropna(axis=1)
        )
        
        div_control_options=st.container()
        with div_control_options:
            
            div_transform_method=st.columns(1)[0]
            with div_transform_method:
                if len(selectedUpper) > 1:
                    transform_methods=list(transform_method_constraint.keys())
                else:
                    transform_methods=[q for q in transform_method_constraint.keys() if q != 'Combat (covariate: gender, age)']
                
                transform_method_selected=st.radio(
                    "Preprocess Method",
                    transform_methods,
                    horizontal=True
                )
                
            
            selected_covariates = []
            if "Combat" in transform_method_selected:
                cov_candidates = [c for c in noe.columns if c.startswith('cov_') or c.lower() == 'age']
                
                default_covs = [c for c in cov_candidates if 'gender' in c.lower() or 'age' in c.lower()]
                
                selected_covariates = st.multiselect(
                    "Select Covariates for ComBat",
                    options=cov_candidates,
                    default=default_covs,
                    format_func=lambda x: x.replace('cov_', '').replace('_', ' ').title() if x.startswith('cov_') else x.title()
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
            noe = deta.transform(
                        noe, 
                        transform_method, 
                        noeVarName, 
                        noeVolName, 
                        covariates=selected_covariates
                    )
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
                noeCatName = [c for c in noeCatName if c.lower() != 'age']
                noeCatName=noeCatName[:-1] if len(selectedUpper)==1 else noeCatName
                lt=st.selectbox(
                    f"Group ({len(noeCatName)})",
                    noeCatName,
                    format_func=lambda x: x.replace('cov_', '').replace('_', ' ').title() if isinstance(x, str) and x.startswith('cov_') else util.sanitise(x),
                    key="l"
                )
            with dropdown_listbox_center:
                noeVolName = sorted(noeVolName)
                ct = st.selectbox(
                    f"X ({len(noeVolName)})",
                    noeVolName,
                    format_func=util.sanitise,
                    key="c"
                )
            with dropdown_listbox_right:
                rt = st.selectbox(
                    f"Y ({len(noeVolName)})",
                    noeVolName,
                    index=min(3, len(noeVolName)-1),
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
    if not data_loaded:
        st.warning("No data to display. Please upload a file.")
    else:
        canvas = st.container(border=True)
        with canvas:
            group_col = Selected.left[0] if hasattr(Selected, 'left') else None
            
            if group_col and group_col in noe.columns:
                try:
                    vs = util.isVs(noe.loc[:, group_col])
                except Exception:
                    st.info("Syncing data context...")
                    st.stop()
                for selected in (Selected.center, Selected.right):
                    method_info = f" ({transform_method_selected})" if transform_method_selected != "None" else ""
                    st.markdown(f"##### 📦 Boxplots<br><sub>Target Feature: {selected[1]}{method_info}</sub>", unsafe_allow_html=True)
                    
                    q, w, e, r = st.columns(4)
                    stats_func = ["mean", "median", "std", "count"]
                    cols = [q, w, e, r]
                    for i, stat in enumerate(stats_func):
                        val = noe[selected[0]].agg(stat)
                        cols[i].metric(stat.title(), f"{val:.2f}" if stat != "count" else f"{val:0}")
                    
                    boxplotLayoutProportion = [7.8, 3.2, .1] if select_switch else 1
                    boxplotImageDivider = st.columns(boxplotLayoutProportion)
                    
                    with boxplotImageDivider[0]:
                        boxplotLeft = util.multiBox(noe, Selected.left, selected, vs=vs)
                        if vs:
                            st.markdown(f"**Statistical Summary**<br><sub>{boxplotLeft[0][1]}, {boxplotLeft[0][2]}</sub>", unsafe_allow_html=True)
                        st.plotly_chart(boxplotLeft[1], use_container_width=True)
                    
                    if select_switch and len(boxplotImageDivider) > 1:
                        with boxplotImageDivider[1]:
                            noeImageEach = util.getNoeImage(noeImage, selected)
                            st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
                            st.image(noeImageEach)

                    if not vs:
                        intergroupTtestResult = util.intergroupTt(noe, Selected.left, selected)
                        st.markdown(f"{intergroupTtestResult[0]}", unsafe_allow_html=True)
                        st.dataframe(intergroupTtestResult[1].style.map(util.sign), use_container_width=True)
                    
                    st.divider()
                try:
                    scatter = util.scatterTrajectory(noe=noe, c=Selected.left, x=Selected.center, y=Selected.right)
                    st.markdown(scatter[0], unsafe_allow_html=True)
                    st.plotly_chart(scatter[1])
                except Exception as err:
                    st.caption("Scatter plot is not available for current selection.")
            else:
                st.error("Selected group column not found in the current dataset.")
                st.stop()

with page_volumetry:
    if not data_loaded:
        st.warning("⚠️ Please upload a dataset in the 'Upload' tab first.")
    else:
        decomposition_vol_name = [q for q in noeVolName if not q.startswith("icv")]

        if not decomposition_vol_name:
            st.error("❌ No valid volume parameters found for analysis.")
            st.stop()

        div_volumetry_plot = st.container(border=True)
        with div_volumetry_plot:
            decomposition_plot_title = f"##### 🔍 Principal Component Analysis<br><sub>{len(decomposition_vol_name)} volume parameters, {transform_method_selected}</sub>"
            st.markdown(decomposition_plot_title, unsafe_allow_html=True)
            
            try:
                decomposed = util.lap(
                    util.decompose,
                    noe=noe,
                    c=Selected.left,
                    y=decomposition_vol_name
                )
                st.pyplot(decomposed[0], use_container_width=True, transparent=True)
                decomposer = decomposed[1]
            except Exception as e:
                st.error(f"PCA generation failed: {e}")

        st.divider()

        st.markdown("##### 📊 Site Effect Evaluation")
        try:
            stats = util.get_site_effect_evaluation(noe, decomposition_vol_name, grouper, covariates=selected_covariates)
            
            if selected_covariates:
                display_covs = [c.replace('cov_', '').replace('_', ' ').title() if c.startswith('cov_') else c.title() for c in selected_covariates]
                st.info(f"**Adjusted for:** {', '.join(display_covs)}")
            else:
                st.caption("ℹ️ No covariates selected for adjustment (Raw Site Effect).")
        
            col1, col2, col3 = st.columns(3)
            col1.metric("Avg. Site p-value", f"{stats['avg_p']:.4f}")
            col2.metric("Significant Features", f"{stats['sig_count']} / {len(decomposition_vol_name)}")
            col3.metric("Mean F-statistic", f"{stats['f_stat']:.2f}")

            if stats['avg_p'] > 0.05:
                st.success("✅ No significant site effect observed (on average).")
            else:
                st.warning("⚠️ Significant site effect remains in some features.")
        except Exception as e:
            st.error("Site effect evaluation failed. Please check your covariates.")

        st.divider()

        st.markdown("##### 📈 Batch Effect Validation via Residuals (Pre vs. Post)")
        display_covs_text = ", ".join([
            c.replace('cov_', '').replace('_', ' ').title() if c.startswith('cov_') else c.title() 
            for c in selected_covariates
            ])
        if "Combat" in transform_method_selected and selected_covariates:
            target_var = st.selectbox(
                "Select Feature to Examine Residuals",
                options=decomposition_vol_name,
                format_func=util.sanitise,
                key="res_comparison_select"
            )
            
            res_comparison_fig = util.draw_residual_comparison(
                noe_before_preprocess, 
                noe, 
                target_var, 
                grouper, 
                selected_covariates
            )
            st.pyplot(res_comparison_fig)
            
            st.info(f"**Adjusted for:** {display_covs_text}  \n**Interpretation:** Post-Combat boxes should be centered around zero.")
        else:
            st.info("💡 To see residual plots, please select **'Combat'** in the sidebar/control panel and specify covariates.")
        
        st.divider()

        main_title = "##### 🎻 Violin Plot"
        cov_text = f" (covariates: {display_covs_text})" if ("Combat" in transform_method_selected and selected_covariates) else ""
        sub_title = f"<sub>{len(decomposition_vol_name)} parameters, {transform_method_selected}{cov_text}</sub>"
        
        st.markdown(f"{main_title}<br>{sub_title}", unsafe_allow_html=True)
        
        violin_fig = util.lap(
            util.draw_violin,
            deta=deta.transform(noe, "scale", noeVarName, decomposition_vol_name),
            value_column=decomposition_vol_name
        )
        st.pyplot(violin_fig[0], use_container_width=True, transparent=True)



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
        if uploaded_file:
            try:
                df_new = pd.read_csv(uploaded_file, na_filter=False)
                save_path = os.path.join(deta.path, uploaded_file.name.replace('.csv', '.f'))
                df_new.to_feather(save_path, compression='zstd', compression_level=9)
                
                st.success(f'Successfully converted: {uploaded_file.name}')
                
                st.cache_data.clear()
                st.session_state['upload_done'] = True
                
                st.rerun()
            except Exception as err:
                st.error(f'File processing error: {err}')


foot=st.container(border=False)
with foot:
    st.markdown(
        """**© 2026 <https://www.nih.go.kr>**""",
        unsafe_allow_html=True
    )

