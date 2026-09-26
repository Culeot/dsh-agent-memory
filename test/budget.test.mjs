// 预算化注入测试 —— "注意力预算"这条核心目标的可验证形式。
//
// 旧规则是"过门槛的都注入,最多 N 条"。它把"一个明确赢家"和"一堆勉强够格的"
// 当成同一回事。预算化选择看的是**分数曲线的形状**:留下第一名,以及确实和它
// 接近的那些。一个答案明显的查询只注入一条,而不是三条平庸的。
//
// 为什么这不是小事:注意力是稀缺资源,注入一条边缘记忆不是中性的 ——
// 它花掉预算,还可能把判断带偏(见 test/degradation、docs/BUILD-PLAN.md)。
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { selectForInjection } from '../lib/index.js';

const R = (score) => ({ score });

describe('预算化注入(注意力预算)', () => {
  it('有明显赢家时只注入一条', () => {
    const out = selectForInjection([R(10), R(3), R(2.9), R(2.8)], { budgetRatio: 0.5, limit: 3 });
    assert.equal(out.length, 1, '离第一名太远的同批记忆不该跟着一起注入');
    assert.equal(out[0].score, 10);
  });

  it('分数接近时保留多条(它们确实都值得看)', () => {
    const out = selectForInjection([R(10), R(9.4), R(8.8), R(4)], { budgetRatio: 0.5, limit: 3 });
    assert.equal(out.length, 3);
    assert.deepEqual(out.map((r) => r.score), [10, 9.4, 8.8]);
  });

  it('budgetRatio=0 关闭预算,退回"门槛 + 上限"的历史行为', () => {
    const out = selectForInjection([R(10), R(3), R(2.9), R(2.8)], { budgetRatio: 0, limit: 3 });
    assert.equal(out.length, 3, '关闭预算时必须与旧行为完全一致(可回退)');
  });

  it('minScore 先过滤,再谈预算', () => {
    const out = selectForInjection([R(10), R(3), R(1)], { minScore: 2, budgetRatio: 0.5, limit: 3 });
    assert.deepEqual(out.map((r) => r.score), [10]);
  });

  it('上限仍然生效(预算是双重约束)', () => {
    const out = selectForInjection([R(10), R(10), R(10), R(10), R(10)], { budgetRatio: 0.5, limit: 2 });
    assert.equal(out.length, 2);
  });

  it('空输入与全零分不炸', () => {
    assert.deepEqual(selectForInjection([], { budgetRatio: 0.5, limit: 3 }), []);
    // top 为 0 时不做比例过滤,否则会把整批合法结果全砍掉
    assert.equal(selectForInjection([R(0), R(0)], { budgetRatio: 0.5, limit: 3 }).length, 2);
  });
});
