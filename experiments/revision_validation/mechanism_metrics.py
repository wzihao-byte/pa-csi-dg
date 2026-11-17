"""Descriptive source-representation measurements; no model selection or claims of causality."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


def fixed_probe_positions(rows, count_per_environment_class=32):
    pools = {}
    for index, row in enumerate(rows):
        key = row["environment"], row["label"]
        digest = hashlib.sha256(("source-environment-probe-v1:"+row["sample_id"]).encode()).hexdigest()
        pools.setdefault(key, []).append((digest, index))
    if len(pools) != 12 or any(len(pool)<count_per_environment_class for pool in pools.values()):
        raise ValueError("Source probe needs 32 examples for each of two environments and six classes")
    return [index for key in sorted(pools) for _,index in sorted(pools[key])[:count_per_environment_class]]


def scatter_statistics(features, labels, environments, epsilon=1e-12):
    x, y, e = np.asarray(features,dtype=np.float64), np.asarray(labels), np.asarray(environments)
    if x.ndim!=2 or len(x)!=len(y) or len(x)!=len(e) or not np.isfinite(x).all() or not len(x):
        raise ValueError("Invalid aligned features, labels or environment IDs")
    center=x.mean(0)
    total=float(np.square(x-center).sum()/len(x))
    between,within,environment_within_class,residual=0.,0.,0.,0.
    domain_distances={}
    for label in np.unique(y):
        mask=y==label; values=x[mask]; domains=e[mask]; class_mean=values.mean(0)
        between+=len(values)*float(np.square(class_mean-center).sum())/len(x)
        within+=float(np.square(values-class_mean).sum())/len(x)
        means=[]
        for domain in np.unique(domains):
            group=values[domains==domain]; mean=group.mean(0); means.append(mean)
            environment_within_class+=len(group)*float(np.square(mean-class_mean).sum())/len(x)
            residual+=float(np.square(group-mean).sum())/len(x)
        pairs=[float(np.square(a-b).sum()) for i,a in enumerate(means) for b in means[i+1:]]
        domain_distances[str(label)]=float(np.mean(pairs)) if pairs else None
    distances=[v for v in domain_distances.values() if v is not None]
    return {"sample_count":len(x),"dimensions":x.shape[1],"class_count":len(np.unique(y)),"environment_count":len(np.unique(e)),
        "total_scatter":total,"between_class_scatter":between,"within_class_scatter":within,
        "between_to_within_class_ratio":between/within if within>epsilon else None,
        "within_class_near_zero":within<=epsilon,"environment_scatter_within_class":environment_within_class,
        "residual_scatter_within_environment_class":residual,
        "class_conditional_domain_squared_distances":domain_distances,
        "equal_class_mean_domain_squared_distance":float(np.mean(distances)) if distances else None,
        "decomposition_total_abs_error":abs(total-between-within),
        "decomposition_within_class_abs_error":abs(within-environment_within_class-residual),"epsilon":epsilon}


def feature_scatter(features, labels, environments):
    x=np.asarray(features,dtype=np.float64); norms=np.linalg.norm(x,axis=1,keepdims=True)
    normalized=np.divide(x,norms,out=np.zeros_like(x),where=norms>1e-12)
    return {"raw_feature":scatter_statistics(x,labels,environments),
            "l2_normalized_feature":scatter_statistics(normalized,labels,environments),
            "near_zero_feature_rows":int((norms<=1e-12).sum())}


def environment_probe(train_features, train_rows, validation_features, validation_rows):
    for key in ("sample_id","subject_id","acquisition_id"):
        if {r[key] for r in train_rows}&{r[key] for r in validation_rows}:
            raise ValueError(f"Environment probe train/validation overlap: {key}")
    train=np.asarray(train_features,dtype=np.float64).copy(); val=np.asarray(validation_features,dtype=np.float64).copy()
    train_labels=np.array([r['label'] for r in train_rows]); val_labels=np.array([r['label'] for r in validation_rows])
    domains=sorted({r['environment'] for r in train_rows})
    if len(domains)!=2 or {r['environment'] for r in validation_rows}!=set(domains):
        raise ValueError('A two-source probe requires both environments in train and validation')
    if set(val_labels)-set(train_labels):
        raise ValueError('A validation class lacks a probe-training centroid')
    # Gold class labels condition this diagnostic. Class centroids and scaling
    # come only from the fixed source-training subset, never the held-out rows.
    centroids={label:train[train_labels==label].mean(0) for label in np.unique(train_labels)}
    for label,center in centroids.items():
        train[train_labels==label]-=center; val[val_labels==label]-=center
    scaler=StandardScaler().fit(train)
    targets=np.array([domains.index(r['environment']) for r in train_rows])
    actual=np.array([domains.index(r['environment']) for r in validation_rows])
    classifier=LogisticRegression(C=1.,solver='lbfgs',max_iter=2000,tol=1e-6,class_weight='balanced',random_state=147)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        classifier.fit(scaler.transform(train),targets)
    predicted=classifier.predict(scaler.transform(val))
    return {'accuracy':float(accuracy_score(actual,predicted)),
        'balanced_accuracy':float(balanced_accuracy_score(actual,predicted)),
        'train_rows':len(train),'validation_rows':len(val),'environments':domains,
        'class_centering_fitted_on_probe_training_only':True,'requires_gold_class_labels':True,
        'scaler_fitted_on_probe_training_only':True,'subject_and_acquisition_disjoint':True,
        'iterations':classifier.n_iter_.tolist(),'converged':not any(issubclass(w.category,ConvergenceWarning) for w in caught),
        'fixed_C':1.,'hyperparameter_search_trials':0,
        'interpretation':'Held-out source environment information accessible to this fixed linear probe; not a proof of target invariance'}, predicted.tolist()


class ResidualObserver:
    """Observe actual conv activations/projection including bias in eval mode."""
    def __init__(self, model, epsilon=1e-8):
        if model.training:
            raise ValueError('Residual observation requires eval mode, where dropout is identity')
        self.epsilon,self.handles,self.records=epsilon,[],{}
        for name,module in model.named_modules():
            if name.endswith(('temporal_pool','channel_pool')):
                if module.__class__.__name__ not in ('IsolatedTemporalPyramidPool','GlobalMlpPool','AttentionResidualPool'):
                    raise ValueError('Unknown pooling implementation; inspect it before interpreting a residual')
                if hasattr(module,'preserve_global') and not module.preserve_global:
                    raise ValueError('This observer requires an actual global residual path')
                self.records[name]=[]; captured={}
                def before(m,inputs,cache=captured):
                    cache.clear();cache['maxima']=[]
                def activation(m,inputs,result,cache=captured):
                    cache['maxima'].append(result.detach().amax(dim=-1))
                def projection(m,inputs,result,cache=captured):
                    cache['projection']=result.detach()
                def after(m,inputs,result,cache=captured,label=name):
                    global_feature=torch.cat(cache['maxima'],dim=1)
                    residual=m.pyramid_scale*cache['projection']
                    discrepancy=float((result.detach()-global_feature-residual).abs().max())
                    # Addition then subtraction introduces normal rounding, so
                    # also compare the exact forward expression directly.
                    forward_error=float((result.detach()-(global_feature+residual)).abs().max())
                    if forward_error>1e-5:
                        raise RuntimeError('Observed components do not reproduce the actual residual forward')
                    g=global_feature.double().norm(dim=1).cpu().numpy(); r=residual.double().norm(dim=1).cpu().numpy()
                    self.records[label].append({'global_norm':g,'residual_norm':r,'ratio':r/(g+self.epsilon),
                        'near_zero_global':g<=self.epsilon,'forward_expression_max_abs_error':forward_error,
                        'subtract_reconstruction_max_abs_error':discrepancy,'scale':m.pyramid_scale})
                self.handles += [module.register_forward_pre_hook(before),module.activation.register_forward_hook(activation),
                                 module.projection.register_forward_hook(projection),module.register_forward_hook(after)]
        if not self.records:
            raise ValueError('No supported residual pooling modules found')

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]


def gradient_alignment(model, primary, regularizer, weight):
    parameters=[(name,p) for name,p in model.named_parameters() if p.requires_grad and not name.startswith(('classifier.','projection_head.'))]
    if not regularizer.requires_grad or weight<=0:
        return {'status':'regularizer_not_active'}
    first=torch.autograd.grad(primary,[p for _,p in parameters],retain_graph=True,allow_unused=True)
    second=torch.autograd.grad(regularizer,[p for _,p in parameters],retain_graph=True,allow_unused=True)
    groups={'all_shared':list(range(len(parameters)))}
    for prefix in ('encoder.position_encoding.','encoder.temporal_encoder.','encoder.temporal_pool.',
                   'encoder.channel_encoder.','encoder.channel_pool.','representation.'):
        groups[prefix.rstrip('.')]=[i for i,(name,_) in enumerate(parameters) if name.startswith(prefix)]
    results={}
    for name,positions in groups.items():
        dot,norm_a,norm_b=0.,0.,0.
        for index in positions:
            a,b=first[index],second[index]
            if a is not None:norm_a+=float(a.double().square().sum())
            if b is not None:norm_b+=float(b.double().square().sum())
            if a is not None and b is not None:dot+=float((a.double()*b.double()).sum())
        a,b=np.sqrt(norm_a),np.sqrt(norm_b)
        results[name]={'primary_gradient_norm':float(a),'regularizer_gradient_norm':float(b),
            'cosine':float(np.clip(dot/(a*b),-1,1)) if a>1e-12 and b>1e-12 else None,
            'weighted_regularizer_to_primary_norm_ratio':float(weight*b/a) if a>1e-12 else None,
            'regularizer_weight':float(weight),'parameter_count':sum(parameters[i][1].numel() for i in positions)}
    return results
