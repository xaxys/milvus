// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"context"

	"github.com/cockroachdb/errors"

	"github.com/milvus-io/milvus/internal/allocator"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// ioErrSentinels is the merr IO family. A file decoder can surface one of these
// through its own error chain -- npyio propagates the reader's error verbatim and
// Arrow wraps it with %w -- and the code has to survive the wrap: ErrImportFailed
// is 2100/InputError, so folding a transient outage into it reports the outage as
// bad input data and makes retry.Do give up (pkg/util/retry/retry.go:86).
//
// Enumerated rather than matched on the 1000-1012 code range, which also holds
// ErrSerializationFailed, ErrStorage and ErrDataIntegrity.
var ioErrSentinels = []error{
	merr.ErrIoKeyNotFound, merr.ErrIoFailed, merr.ErrIoUnexpectEOF,
	merr.ErrIoTooManyRequests, merr.ErrIoPermissionDenied, merr.ErrIoBucketNotFound,
	merr.ErrIoInvalidCredentials, merr.ErrIoInvalidArgument, merr.ErrIoInvalidRange,
	merr.ErrIoEntityTooLarge,
}

// IsTypedIOErr reports whether err carries a merr IO code anywhere in its chain.
func IsTypedIOErr(err error) bool {
	for _, sentinel := range ioErrSentinels {
		if errors.Is(err, sentinel) {
			return true
		}
	}
	return false
}

// WrapDecodeErr classifies a failure raised while decoding an import file.
//
// A reader in front of the decoder retries transient object-store faults, but
// every such layer gives up after a bounded number of attempts -- and the parquet
// footer is read through ReadAt/Seek, which retryableReader does not even override.
// So a persistent outage does reach the decoder. Keep a typed IO cause and its
// System classification; report anything else -- a real decode failure, which
// carries no merr code -- as input.
//
// UPGRADE CHECK -- this classification depends on the decoder preserving the
// cause. IsTypedIOErr walks the chain with errors.Is, so a decoder that
// stringifies its cause instead of wrapping it would turn a persistent storage
// outage into an InputError and abort the caller's retry.Do. Verified against
// the pinned versions: arrow-go v17 wraps with %w throughout
// parquet/file/file_reader.go, and npyio v0.6.0 returns reader errors unmodified.
// Re-verify both when either dependency is bumped; TestWrapDecodeErr covers the
// wrapping semantics but cannot detect a dependency that stops wrapping.
func WrapDecodeErr(err error, what string) error {
	if IsTypedIOErr(err) {
		return merr.Wrap(err, what)
	}
	return merr.WrapErrImportFailedMsg("%s, err=%v", what, err)
}

// The Import V3 verdict is stage-specific: an error that is terminal for a
// task execution can be transient for a Create RPC (whose outcome is unknown)
// or for an acceptResult poll (catalog/transport can recover). Each predicate
// therefore uses the same two semantic properties -- InputError and the
// producer's retriable flag -- and a different default for unclassified
// errors. No per-error denylist is needed as long as the storage/FFI boundary
// maps transient failures to merr.ErrStorageTransient (retriable) and
// permanent ones to merr.ErrStorage (or a more specific permanent code).

// isContextTransientNotFound reports errors that mean "metadata has not
// propagated yet" rather than "the object is permanently gone". DataCoord must
// keep retrying these through the transient not-found path; the proxy stamps
// the user-facing variants as InputError before they reach a client.
func isContextTransientNotFound(err error) bool {
	return errors.IsAny(err,
		merr.ErrCollectionNotFound,
		merr.ErrPartitionNotFound,
		merr.ErrDatabaseNotFound,
		merr.ErrSegmentNotFound,
		merr.ErrNodeNotFound)
}

// IsTerminalImportV3TaskErr is the DataNode execution verdict. It is
// default-deny: an unclassified execution error fails the task, while only
// errors explicitly marked retriable (or context-transient not-found) retry.
func IsTerminalImportV3TaskErr(err error) bool {
	if err == nil {
		return false
	}
	if errors.IsAny(err, context.Canceled, context.DeadlineExceeded) {
		return false
	}
	// ID exhaustion is recoverable: DataCoord's prepareRetry re-derives the
	// per-file ID width before the next run.
	if allocator.IsIDExhausted(err) {
		return false
	}
	if merr.GetErrorType(err) == merr.InputError {
		return true
	}
	// A raw error has no classification; the execution default is failure.
	if !merr.IsMilvusError(err) {
		return true
	}
	if isContextTransientNotFound(err) {
		return false
	}
	return !merr.IsRetryableErr(err)
}

// IsTerminalImportV3CreateErr is the DataCoord Create* verdict. It is
// default-retry for raw transport errors (the RPC outcome is unknown) and for
// errors explicitly marked retriable; a typed non-retriable rejection fails.
func IsTerminalImportV3CreateErr(err error) bool {
	if err == nil {
		return false
	}
	if merr.GetErrorType(err) == merr.InputError {
		return true
	}
	if !merr.IsMilvusError(err) {
		return false
	}
	if merr.IsRetryableErr(err) {
		return false
	}
	// The node may have disappeared, or the collection/partition metadata may
	// not have propagated yet; a new run can be routed elsewhere.
	if isContextTransientNotFound(err) {
		return false
	}
	return true
}

// IsTerminalImportV3AcceptErr is the DataCoord acceptResult verdict. It is
// default-retry for raw catalog/transport errors and for typed retriable
// errors; a typed non-retriable error (or InputError) fails.
func IsTerminalImportV3AcceptErr(err error) bool {
	if err == nil {
		return false
	}
	if merr.GetErrorType(err) == merr.InputError {
		return true
	}
	if !merr.IsMilvusError(err) {
		return false
	}
	if isContextTransientNotFound(err) {
		return false
	}
	return !merr.IsRetryableErr(err)
}

// IsTerminalImportV3JobErr is the DataCoord job-handler verdict. Like
// acceptResult, control-plane failures default to retry; only InputError or a
// typed non-retriable error fails the job.
func IsTerminalImportV3JobErr(err error) bool {
	return IsTerminalImportV3AcceptErr(err)
}
