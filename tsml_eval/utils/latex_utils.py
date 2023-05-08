import numpy as np
import warnings
warnings.filterwarnings('ignore')

latex_str = """
                      Haptics & 0.3831 & 0.3474 & 0.3799 & 0.3506 & 0.2955 & 0.3377 & 0.3214 & 0.3279 & 0.3182 & 0.2922 \\
                       Herring & 0.5469 & 0.5469 & 0.5469 & 0.5312 & 0.5156 & 0.5781 & 0.5469 & 0.5312 & 0.5625 & 0.5156 \\
                   HouseTwenty & 0.8067 & 0.8655 & 0.8571 & 0.7143 & 0.7647 & 0.5882 & 0.5798 & 0.6218 & 0.7815 & 0.8487 \\
                   InlineSkate & 0.2200 & 0.2091 & 0.2673 & 0.2364 & 0.2200 & 0.2218 & 0.2873 & 0.2618 & 0.2327 & 0.2545 \\
         InsectEPGRegularTrain & 0.4940 & 0.5181 & 0.5663 & 0.5823 & 0.6024 & 0.5823 & 0.7149 & 0.5542 & 0.6265 & 0.5823 \\
           InsectEPGSmallTrain & 0.6024 & 0.6345 & 0.6265 & 0.4498 & 0.6185 & 0.5462 & 0.5020 & 0.4980 & 0.6466 & 0.6867 \\
           InsectWingbeatSound & 0.4278 & 0.4253 & 0.3202 & 0.3263 & 0.2707 & 0.4394 & 0.1742 & 0.3551 & 0.2424 & 0.2217 \\
              ItalyPowerDemand & 0.5190 & 0.5287 & 0.5471 & 0.5306 & 0.5335 & 0.5170 & 0.9252 & 0.9310 & 0.6793 & 0.5005 \\
        LargeKitchenAppliances & 0.5707 & 0.5573 & 0.4827 & 0.4453 & 0.4053 & 0.4373 & 0.6240 & 0.3867 & 0.4693 & 0.4853 \\
                    Lightning2 & 0.5082 & 0.6393 & 0.7049 & 0.6721 & 0.6721 & 0.6393 & 0.5902 & 0.5738 & 0.5410 & 0.5574 \\
                    Lightning7 & 0.5205 & 0.5616 & 0.4932 & 0.6164 & 0.5205 & 0.4110 & 0.4247 & 0.4795 & 0.4521 & 0.4521 \\
                        Mallat & 0.8286 & 0.8026 & 0.7923 & 0.8196 & 0.7953 & 0.6917 & 0.5100 & 0.5898 & 0.4857 & 0.4072 \\
                          Meat & 0.6833 & 0.6833 & 0.6833 & 0.7000 & 0.8000 & 0.7333 & 0.4667 & 0.5167 & 0.6333 & 0.6167 \\
                 MedicalImages & 0.2868 & 0.3013 & 0.3684 & 0.3566 & 0.3184 & 0.2776 & 0.2711 & 0.2776 & 0.2382 & 0.2237 \\
  MiddlePhalanxOutlineAgeGroup & 0.4740 & 0.4935 & 0.4805 & 0.5844 & 0.4610 & 0.4805 & 0.4675 & 0.4675 & 0.5974 & 0.5974 \\
   MiddlePhalanxOutlineCorrect & 0.5945 & 0.5945 & 0.5979 & 0.5876 & 0.5876 & 0.6151 & 0.5842 & 0.5842 & 0.5739 & 0.5876 \\
               MiddlePhalanxTW & 0.4026 & 0.4416 & 0.4481 & 0.4675 & 0.4545 & 0.4221 & 0.5390 & 0.4935 & 0.3701 & 0.3636 \\
       MixedShapesRegularTrain & 0.6507 & 0.6515 & 0.5761 & 0.4569 & 0.6544 & 0.6359 & 0.6194 & 0.4911 & 0.5369 & 0.5728 \\
         MixedShapesSmallTrain & 0.6301 & 0.6120 & 0.5901 & 0.5963 & 0.5518 & 0.6474 & 0.6313 & 0.6602 & 0.6247 & 0.6033 \\
                    MoteStrain & 0.8738 & 0.8850 & 0.8826 & 0.5543 & 0.8610 & 0.7971 & 0.5591 & 0.5495 & 0.8490 & 0.7995 \\
    NonInvasiveFetalECGThorax1 & 0.4270 & 0.4132 & 0.4163 & 0.3573 & 0.3766 & 0.4061 & 0.2967 & 0.2443 & 0.3277 & 0.3552 \\
    NonInvasiveFetalECGThorax2 & 0.5079 & 0.4712 & 0.4830 & 0.4539 & 0.4987 & 0.4234 & 0.3522 & 0.2295 & 0.3883 & 0.4453 \\
                      OliveOil & 0.7333 & 0.7333 & 0.7333 & 0.7333 & 0.7667 & 0.7333 & 0.4000 & 0.4000 & 0.5333 & 0.6333 \\
                       OSULeaf & 0.4008 & 0.5124 & 0.3678 & 0.3636 & 0.3554 & 0.3347 & 0.5413 & 0.5537 & 0.4091 & 0.3554 \\
      PhalangesOutlinesCorrect & 0.6329 & 0.6212 & 0.6282 & 0.6247 & 0.6247 & 0.6282 & 0.6014 & 0.6247 & 0.5874 & 0.6282 \\
                       Phoneme & 0.2015 & 0.1345 & 0.1783 & 0.1630 & 0.1762 & 0.0992 & 0.1883 & 0.1967 & 0.1857 & 0.1698 \\
             PigAirwayPressure & 0.1779 & 0.1923 & 0.1683 & 0.1779 & 0.1442 & 0.1875 & 0.2500 & 0.2212 & 0.1827 & 0.1683 \\
                PigArtPressure & 0.2548 & 0.3413 & 0.1827 & 0.2692 & 0.2212 & 0.3510 & 0.3077 & 0.2260 & 0.2404 & 0.2837 \\
                        PigCVP & 0.2596 & 0.2115 & 0.1875 & 0.2260 & 0.1587 & 0.2260 & 0.1923 & 0.1779 & 0.2356 & 0.3269 \\
                         Plane & 0.8000 & 0.8286 & 0.8857 & 0.8571 & 0.7619 & 0.9238 & 0.7619 & 0.8000 & 0.8381 & 0.9429 \\
                     PowerCons & 0.7389 & 0.7889 & 0.7333 & 0.6944 & 0.5944 & 0.8278 & 0.5778 & 0.6611 & 0.7000 & 0.5889 \\
ProximalPhalanxOutlineAgeGroup & 0.8000 & 0.7902 & 0.8000 & 0.7854 & 0.7902 & 0.7317 & 0.7951 & 0.7951 & 0.7561 & 0.7707 \\
 ProximalPhalanxOutlineCorrect & 0.6495 & 0.6495 & 0.6460 & 0.6426 & 0.6426 & 0.6460 & 0.6357 & 0.6357 & 0.6323 & 0.6426 \\
             ProximalPhalanxTW & 0.5268 & 0.5122 & 0.4829 & 0.5512 & 0.5415 & 0.5024 & 0.5220 & 0.4927 & 0.5122 & 0.4585 \\
          RefrigerationDevices & 0.4960 & 0.4560 & 0.5387 & 0.4560 & 0.5467 & 0.3627 & 0.4187 & 0.3947 & 0.4373 & 0.4293 \\
                          Rock & 0.5800 & 0.4600 & 0.3400 & 0.6200 & 0.4200 & 0.4400 & 0.5000 & 0.5200 & 0.4600 & 0.5600 \\
                    ScreenType & 0.4187 & 0.4213 & 0.4053 & 0.3920 & 0.3973 & 0.3973 & 0.3520 & 0.4587 & 0.3760 & 0.3680 \\
             SemgHandGenderCh2 & 0.5683 & 0.6667 & 0.6517 & 0.6150 & 0.6733 & 0.6183 & 0.6833 & 0.6500 & 0.6667 & 0.6733 \\
           SemgHandMovementCh2 & 0.3400 & 0.3089 & 0.2911 & 0.3444 & 0.3289 & 0.2933 & 0.2889 & 0.2911 & 0.3267 & 0.2533 \\
            SemgHandSubjectCh2 & 0.4667 & 0.5356 & 0.4400 & 0.4756 & 0.4422 & 0.4267 & 0.4200 & 0.4178 & 0.3667 & 0.3400 \\
                   ShapeletSim & 0.6444 & 0.6278 & 0.6278 & 0.5667 & 0.5333 & 0.5111 & 0.5222 & 0.5944 & 0.5556 & 0.6444 \\
                     ShapesAll & 0.4067 & 0.4283 & 0.2433 & 0.3300 & 0.2367 & 0.4117 & 0.2350 & 0.2517 & 0.2950 & 0.3900 \\
        SmallKitchenAppliances & 0.6293 & 0.6400 & 0.6960 & 0.4480 & 0.6853 & 0.3787 & 0.6320 & 0.4000 & 0.5947 & 0.5813 \\
                SmoothSubspace & 0.7267 & 0.8267 & 0.7467 & 0.7400 & 0.7400 & 0.5933 & 0.4600 & 0.4667 & 0.3733 & 0.4000 \\
         SonyAIBORobotSurface1 & 0.6057 & 0.5641 & 0.6722 & 0.5125 & 0.6755 & 0.5724 & 0.9168 & 0.6972 & 0.5774 & 0.5624 \\
         SonyAIBORobotSurface2 & 0.7702 & 0.7880 & 0.8132 & 0.6905 & 0.6695 & 0.7450 & 0.8006 & 0.8048 & 0.6317 & 0.6590 \\
               StarLightCurves & 0.7528 & 0.7640 & 0.7657 & 0.7660 & 0.7329 & 0.7550 & 0.5918 & 0.6474 & 0.7552 & 0.7489 \\
                    Strawberry & 0.5730 & 0.5730 & 0.5486 & 0.5027 & 0.5027 & 0.5432 & 0.6081 & 0.6027 & 0.5541 & 0.5541 \\
                   SwedishLeaf & 0.5136 & 0.5616 & 0.4912 & 0.3856 & 0.3696 & 0.4496 & 0.5040 & 0.4832 & 0.2352 & 0.2864 \\
                       Symbols & 0.9256 & 0.9477 & 0.8101 & 0.7518 & 0.8281 & 0.7256 & 0.4281 & 0.3709 & 0.7789 & 0.7779 \\
              SyntheticControl & 0.6600 & 0.7133 & 0.6133 & 0.7300 & 0.8000 & 0.4233 & 0.3867 & 0.3200 & 0.4167 & 0.5033 \\
              ToeSegmentation1 & 0.5219 & 0.5044 & 0.5307 & 0.5175 & 0.5307 & 0.5088 & 0.5307 & 0.5439 & 0.5746 & 0.5570 \\
              ToeSegmentation2 & 0.5154 & 0.5615 & 0.7692 & 0.5308 & 0.6308 & 0.5692 & 0.7000 & 0.7615 & 0.6769 & 0.6538 \\
                         Trace & 0.5500 & 0.5600 & 0.5600 & 0.7800 & 0.7600 & 0.5700 & 0.7500 & 0.7300 & 0.5400 & 0.7200 \\
                    TwoLeadECG & 0.7226 & 0.7498 & 0.6471 & 0.6005 & 0.5961 & 0.5391 & 0.7480 & 0.7533 & 0.7568 & 0.7954 \\
                   TwoPatterns & 0.3945 & 0.3928 & 1.0000 & 0.4912 & 0.5178 & 0.2968 & 0.6182 & 0.4270 & 0.3132 & 0.2892 \\
                           UMD & 0.5069 & 0.5000 & 0.4861 & 0.5833 & 0.5833 & 0.4514 & 0.6042 & 0.5347 & 0.5556 & 0.5139 \\
        UWaveGestureLibraryAll & 0.6993 & 0.5265 & 0.5078 & 0.6901 & 0.4883 & 0.7021 & 0.3881 & 0.3763 & 0.2432 & 0.2909 \\
          UWaveGestureLibraryX & 0.6080 & 0.5237 & 0.5073 & 0.6309 & 0.5025 & 0.5022 & 0.2312 & 0.2253 & 0.4531 & 0.2702 \\
          UWaveGestureLibraryY & 0.4838 & 0.4545 & 0.3599 & 0.4757 & 0.4618 & 0.4877 & 0.2164 & 0.1988 & 0.3557 & 0.3029 \\
          UWaveGestureLibraryZ & 0.5039 & 0.4997 & 0.5050 & 0.4863 & 0.4598 & 0.4740 & 0.1762 & 0.2208 & 0.3903 & 0.2987 \\
                         Wafer & 0.6277 & 0.6335 & 0.6288 & 0.6272 & 0.6274 & 0.6308 & 0.7961 & 0.7907 & 0.6048 & 0.5031 \\
                          Wine & 0.5185 & 0.5185 & 0.5185 & 0.5185 & 0.5185 & 0.5185 & 0.5556 & 0.5556 & 0.5185 & 0.5185 \\
                  WordSynonyms & 0.3245 & 0.3009 & 0.3166 & 0.3668 & 0.3495 & 0.2868 & 0.2900 & 0.2900 & 0.2571 & 0.2163 \\
                         Worms & 0.4286 & 0.4935 & 0.4286 & 0.4545 & 0.3766 & 0.2857 & 0.5584 & 0.4026 & 0.4156 & 0.4416 \\
                 WormsTwoClass & 0.5195 & 0.5714 & 0.5065 & 0.5195 & 0.5325 & 0.5325 & 0.5455 & 0.5584 & 0.5974 & 0.5325 \\
                          Yoga & 0.5020 & 0.5013 & 0.5340 & 0.5423 & 0.5007 & 0.5107 & 0.5720 & 0.5490 & 0.5273 & 0.5497 \\
"""

def _iter_rows_and_clean(latex_table_str: str):
    latex_split = latex_table_str.split('\n')

    for line in latex_split:
        line = line.replace('\textbf{', '')
        line = line.replace('}', '')
        line = line.replace(' ', '')
        line = line.replace('\\', '')
        if '&' not in line:
            continue
        split_values = line.split('&')
        yield split_values

def _latex_str_to_pandas(latex_table_str: str, ignore_row_first_value = False, bold_min_value = False, bold_max_value = False, bold_by_col = False, bold_by_row = False):
    import pandas as pd
    df = pd.DataFrame()
    starting_index = 0
    if ignore_row_first_value:
        starting_index = 1

    first_col_included = pd.DataFrame()

    for line in _iter_rows_and_clean(latex_table_str):
        df = df.append(pd.Series(line[starting_index:]), ignore_index=True)
        first_col_included = first_col_included.append(pd.Series(line[0:]), ignore_index=True)

    if bold_by_col:
        if bold_min_value:
            bold_df = df.apply(lambda x: x == x.min(), axis=0)
        else:
            bold_df = df.apply(lambda x: x == x.max(), axis=0)
    else:
        if bold_min_value:
            bold_df = df.apply(lambda x: x == x.min(), axis=1)
        else:
            bold_df = df.apply(lambda x: x == x.max(), axis=1)

    bolded_df = df.mask(bold_df, '\\textbf{' + df.astype(str) + '}')

    i = 0
    for row in bolded_df.values:
        # get name i from first column of df
        print(first_col_included.iloc[i, 0] + ' & ' + ' & '.join(row) + '\\\\')
        i += 1

if __name__ == '__main__':
    # bold_table_by_row(latex_str, find_min_value=True)
    import pandas as pd

    table_data = pd.read_csv('/home/chris/Downloads/TESTCL-ACC_MEANS.csv')
    print(table_data)
    print(table_data.round(4).to_latex(index=False))
    av_column = table_data.mean(axis=0)
    print(max(av_column.values))
    print(av_column)

    string_ver = ''
    arr = []
    for val in av_column.values:
        string_ver += str(round(val, 4)) + ' & '
        arr.append(round(val, 4))

    print(string_ver)

    print(max(arr))



    # _latex_str_to_pandas(latex_str, bold_by_row=True, bold_max_value=True, ignore_row_first_value=True)
