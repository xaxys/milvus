package rootcoord

import (
	"context"
	"errors"
	"testing"

	"github.com/milvus-io/milvus/internal/util/retry"

	etcdkv "github.com/milvus-io/milvus/internal/kv/etcd"

	"github.com/stretchr/testify/assert"

	"github.com/milvus-io/milvus/internal/util/etcd"
)

func Test_etcd(t *testing.T) {
	Params.Init()
	etcdCli, err := etcd.GetEtcdClient(&Params.EtcdCfg)
	assert.NoError(t, err)

	txn := etcdkv.NewEtcdKV(etcdCli, "by-dev")
	err = txn.Save("key", "value")
	assert.NoError(t, err)
	err = txn.Remove("key")
	assert.NoError(t, err)
}

func Test_retry(t *testing.T) {
	f := func() error {
		return errors.New("mock")
	}
	err := retry.Do(context.Background(), f)
	assert.Error(t, err)
}
