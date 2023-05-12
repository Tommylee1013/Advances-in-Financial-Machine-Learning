# Financial Machine Learning
*AI in financial Economics Capstone Project Page*

Marcos Lopez De Prado의 Advances in Financial Machine Learning을 참조하여 제작하였습니다<br>
서강대학교 경제학부 AI Finance의 프로젝트 페이지 입니다 <br>
개인 사용 및 프로젝트 목적으로 만들었으므로 틀린 부분이 있을 수 있습니다

### Introduce FinancialMachineLearning Library
주요 함수에 대해 설명합니다

**Bar Sampling**<br>
`BarSampling` 함수를 사용해 간편하게 Sampling이 가능합니다

```angular2html
dollar_df = fml.BarSampling(df, 'dv', dollar_M)
```

**Fractionally Differencing**<br>
`fracDiff` 함수를 사용하여 분수 미분이 가능합니다

```angular2html
frac_diff = fml.fracDiff_FFD(dollar_df, min_ffd, thres = 1e-5)
```

**CUSUM Filtering**<br>
`getTEvents` 함수를 이용해 fractionally Differentiated Dollar bar price 계열로부터 이벤트 추출 Sampling이 가능합니다

```angular2html
tEvents = fml.getTEvents(frac_diff.price, h = dfx2.std().iat[0] * 2)
```

**Rolling Volatility**<br>
getDailyVolatility 함수를 사용하여 지정된 기간동안의 변동성 추정이 가능합니다(default = 100)
```angular2html
dailyVol = fml.getDailyVolatility(dollar_df)
```

**Vertical Barrier**<br>
`addVerticalBarrier` 함수를 이용하여 수직 배리어 구축이 가능합니다
```angular2html
t1 = fml.addVerticalBarrier(tEvents, dollar_df, numDays = 5)
```

**Concurrency**<br>
`getConcurrentBar` : 중첩 레이블 상태에 있는 Bar 추출 가능<br>
`getAvgLabelUniq` : Label의 평균 고유도를 추정<br>
`mpPandasObj`를 통해서 python MultiProcessing이 가능

```angular2html
numCoEvents = fml.mpPandasObj(fml.getConcurrentBar, ('molecule', events.index), cpus, closeIdx = feature_Mat.index, t1 = events['t1'])
numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep = 'last')]
numCoEvents = numCoEvents.reindex(feature_Mat.index).fillna(0)
out = pd.DataFrame()
out['tW'] = fml.mpPandasObj(fml.getAvgLabelUniq, ('molecule', events.index), cpus, t1 = events['t1'], numCoEvents = numCoEvents)
```

**Cross Validation Score**<br>
`cvScore`함수를 통해 순차적 데이터에 대한 교차 검증 점수를 계산할 수 있습니다

```angular2html
rf = RandomForestClassifier(n_estimators = 1000, criterion = "entropy", bootstrap = True,
                                n_jobs=1, random_state=42, class_weight = 'balanced_subsample', oob_score=False)
cv_gen = KFold(n_splits = 10, shuffle = False)
score = fml.cvScore(rf, X, y, sample_weight = cweight, scoring = 'neg_log_loss', cv = None, cvGen = cv_gen, pctEmbargo = 0)
```
단, 여기서 사용되는 sample_weight는 행렬 X,y와 길이가 같아야 합니다. 또한, pandas version은 1.5.3으로 맞춰야 합니다