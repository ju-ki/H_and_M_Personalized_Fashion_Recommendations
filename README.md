# H_and_M_Personalized_Fashion_Recommendations

## Description
H&Mグループは、53のオンラインマーケットと約4,850の店舗を持つ、ブランドとビジネスのファミリーです。オンラインストアでは、買い物客に豊富な品揃えの中から好きなものを選んでもらうことができます。しかし、選択肢が多すぎると、お客様は興味のあるものや探しているものをすぐに見つけることができず、最終的に購入に至らないこともあります。ショッピング体験を向上させるためには、商品のレコメンデーションが重要な鍵となります。さらに重要なことは、お客様の正しい選択を支援することは、返品を減らし、それによって輸送に伴う排出を最小限に抑えるという、サステナビリティにも良い影響を与えるということです。

このコンペティションでは、H&Mグループが、過去の取引データ、および顧客と商品のメタデータに基づく商品推奨を開発するよう、皆さんに呼びかけます。メタデータは、衣服の種類やお客様の年齢などの単純なデータから、商品説明のテキストデータ、衣服画像の画像データまで幅広く用意されています。

どのような情報が有用であるかという先入観はなく、それはお客様自身が調べることです。カテゴリーデータ型のアルゴリズムを調べたり、NLPや画像処理のディープラーニングに飛び込んでみたり、それはあなた次第です。

## Evaluation

投稿は、平均平均精度@12（MAP@12）に従って評価されます。

<img src="https://latex.codecogs.com/svg.image?MAP@12=\frac{1}{U}\sum_{u=1}^{U}\sum_{k=1}^{min(n,12)}P(k)&space;*&space;rel(k)" title="MAP@12=\frac{1}{U}\sum_{u=1}^{U}\sum_{k=1}^{min(n,12)}P(k) * rel(k)" />
ここで、𝑈U は顧客の数、𝑃(𝑘)P(k) はカットオフ𝑘kにおける精度、𝑛n は画像あたりの予測数、𝑟𝑒𝑙(𝑘)rel(k) はランク𝑘kの項目が関連（正しい）ラベルであれば1、それ以外は0となる指標関数のことである。

注意事項

学習データで購入したかどうかに関わらず、提供されたすべての customer_id 値に対して購入予測を行うことになります。
テスト期間中に購入を行わなかった顧客はスコアリングから除外されます。
12個以下の商品を注文した顧客に対して、12個の予測をフルに使用しても決してペナルティはありません；したがって、各顧客に対して12個の予測をすることが有利です。
提出ファイル
学習データで観測された各 customer_id に対して、article_id（学習時間帯の次の7日間に顧客が購入する予測アイテム）のラベルを最大12個まで予測することができます。ファイルはヘッダーを含み、以下のフォーマットである必要があります。

## Data
この課題では、お客様の購入履歴とそれを裏付けるメタデータが与えられます。あなたの課題は、学習データが終了した直後の7日間に、各顧客がどのような記事を購入するかを予測することです。その間に何も購入しなかった顧客は、スコアリングから除外される。

#### ファイル
* images/ - 各 article_id に対応する画像のフォルダ。画像は article_id の最初の 3 桁で始まるサブフォルダに配置されます。
* articles.csv - 購入可能な各 article_id の詳細なメタデータ。
* customers.csv - データセットに含まれる各顧客IDのメタデータ
* sample_submission.csv - 正しい形式の投稿ファイルのサンプル
* transactions_train.csv - 学習用データで、各顧客の各日付の購入品と追加情報で構成されています。重複する行は、同じ商品の複数購入に対応します。あなたのタスクは、各顧客がトレーニングデータ期間の直後の7日間に購入するarticle_idを予測することです。
* 注：サンプル送信で見つかったすべての customer_id 値について予測を行う必要があります。テスト期間中に購入したすべての顧客は、トレーニングデータで購入履歴があるかどうかに関係なく、スコアリングされます。

## LeaderBoard
このリーダーボードは、約1％のテストデータで計算されています。最終的には残りの99％のデータを元に算出されますので、最終的な順位は異なる場合があります。

### 2022/02/08
* kaggle apiコマンドもしかしたら複数データ(画像など)に対応していない。(途中many requestとか出て打ち切られる)
* 画像以外はcolabでやっていく
* 何となく雰囲気は掴めた。だが学習の仕方とかも含めベースラインが全然見当つかない
* 軽めのEDAをやった.かなり自由なのでアイデア勝負とかになりそう???
* pd.mergeとpd.joinだとjoinの方が速いことを知った。
* csv遅い,pickle速い
* my_pipelineのnlp可視化に関してNaNに対応できていないことが発覚
* 当たり前かもしれないが商品説明は生地に関する単語が多かった(softやbackなど)
* 一旦評価指標の理解とshopeeなどを見て今後のアプローチにについて考える

### 2022/02/09
* とりあえず公開ノートブックを参考にベースラインを作成してみる
* どうやらtransaction_train.csvをそのまま読み込むとarticle_idの先頭の0が読み込めずスコアが可笑しくなるとのこと->read_csvの際にstrしてしてあげる
* ベースラインを作成したことで何となくだけど方向性掴めた気がする(スコア上げれるとは言ってない)
* 上記のことを考慮して再度データセットの作成
* EDAがかなり重要そうになってきそう
* trainの期間にいない人は9699人のデータ->他の購買行動が似ている人の類似度を求めるとか？
* 母親曰く2年で23点は割とそうなのではと返ってきた

### 2022/02/12
* 推薦システムがどのような感じで構成されているのかを確かめる
* 強化学習がなぜディスカッションに出てきたか->ある環境での収益の最大化をするような学習するのが強化学習の特徴->必ずしも人間の教示は必要ないのでは?
* wantedlyさんの輪読会についてのリポジトリパッとみかなり情報ありそう.(今回のコンペ以外でもかなり役に立ちそうなので注目していく)
* trainにはない顧客をどうするか->新規顧客をどうすればいいかと考える必要がある??
* ざっくりと他の企業のレコメンドとか見た->サイトの情報とか(クリック率をどう置き換えるか??)
* 最終スコア60~70%位か(shoppeで78％くらい)->マッチングとレコメンドどっちが難しいかと考えた時直感的にはレコメンド(あくまでリークとかない前提)

### 2022/02/13
* Implicitとかいうライブラリを使ってみる(ALSとかいう推薦向けのライブラリ？)
* 他にscikit-surprise, fast_aiとかがありそう
* cvの算出方法, oof(必要かわからないけど)は全く見当がつかない(恐らく公開ノートでると思うけど)
* まだ気にする必要はないけどアンサンブルってどうやるんだ(argmax位しか思いつかない)
* exp1:一番最小のパラメータでやったら0.006位でた
* ハイパラチューニングしたやつは現在回している
* exp3:期間の設定('2020-08-21')より上のみで学習->0.006

### 2022/02/14
* exp2:ハイパラチューニング的なやつをした全期間学習->0.006位
* exp4:チューニングした限定期間->0.014

### 2022/02/23
* とりあえず公開ノートブックが今何をやってるかを理解した?
* 最も人気の商品top12をそのまま選ぶタイプ
* ある特定の期間で買われている商品を選ぶタイプ
* 意外と頻繁にサブされているわけではない->画像や自然言語などのディープラーニングなどはまだ使用されていないと予想
* シンプルなpandasやEDAでそのスコアをとっているのでは？

### 2022/02/24
* validationを追加してやってみた(CV:0.0235) -> LB:0.010???
* validation+期間を増やす+exponetial decay ->LB:0.019
* この差が何なのか分からない(どうして最後期間を増やしたのかもちょっと分からない)