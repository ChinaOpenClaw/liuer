# 人格档案目录

## 隐私优先

档案用于"以某个真实原型建模"的场景，可能包含出生年份、家庭情况、
健康与经济状况、以及这个人容易被说动的地方——都属于敏感个人信息。

**默认规则：除 `template.yaml` 与本文件外，目录下所有 `*.yaml` 都不进版本库**
（见根目录 `.gitignore`）。

把档案分享给别人之前，请务必：

- 删除或替换真实姓名、具体单位、住址、银行
- 确认 `based_on_real_person` 标记与实际一致
- 想清楚这些信息一旦公开无法撤回

## 用法

```bash
# 用指定档案启动
python -m dolos --profile profiles/你的档案.yaml

# 或在 config.yaml 里固定
# persona:
#   profile: profiles/你的档案.yaml
```

复制模板开始：

```bash
cp profiles/template.yaml profiles/我的档案.yaml
```

## 字段说明

| 字段 | 作用 | 对应模态 |
|---|---|---|
| `context_profile` | 时代背景、这一代人的共同经验 | 语境 |
| `personal_situation` | 家庭、健康、经济状况 | 个人情况 |
| `reflexive_personality` | 自我反思方式 | 反身性格 / 反身性 |
| `empathy_style` | 表达理解的方式 | 同理心 |
| `comfort_style` | 关心他人的方式 | 爱心安抚 |
| `patience_style` | 不急不躁的表现 | 耐心 |
| `reasoning_style` | 讲道理、较真的方式 | 诡辩 |
| `elicit_style` | 追问细节时的口吻 | 套话 |
| `susceptibility` | 易被说动的点 | 策略预判 |

## 写档案的两个要点

**一、`reflexive_personality` 是最关键的一条。** 受过良好教育的人，
怀疑永远指向自己而不是对方——"我是不是没听懂""是不是我落伍了"。
把这条写实，人格才有说服力；写反了（让角色去质疑对方）会立刻暴露。

**二、`susceptibility` 要写具体套路，不要写"容易上当"。**
"自称官方的来电"能用，"容易上当"没用——前者策略层可以据此预判，后者不能。
